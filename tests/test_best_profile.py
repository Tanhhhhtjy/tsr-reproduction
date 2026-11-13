"""Best-policy integration and native inference regressions; no GPU or API calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_training_config import configure,NAMES


def record(responses,scores,prompt='q'):
    return dict(prompt=prompt,responses=responses,scores=scores,
                judgments=[[f'Score: {value}'] for value in scores])


class BestPairIntegration(unittest.TestCase):
    def build(self,current,past,future=None):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);mode='past' if future is None else 'future'
        command=[sys.executable,str(ROOT/f'settings/tsr/reformat2DPO_{mode}.py'),
                 '--best','--output_path',str(root/'output/train.json')]
        for label,row in [('chosen',current),('rejected',past),('future',future)]:
            if row is not None:
                path=root/(label+'.json');path.write_text(json.dumps([row]))
                command.extend(['--input_path_'+label,str(path)])
        subprocess.run(command,check=True,capture_output=True)
        rows=[json.loads(line) for line in (root/'output/train.json').read_text().splitlines()]
        stats=json.loads((root/'output/stats.json').read_text())
        return rows,stats

    def test_future_and_current_source_priorities_precede_character_cap(self):
        current=record(['short current best','current bad'],[5,0])
        past=record(['past bad'],[0]);future=record(['f'*2501,'eligible future'],[5,5])
        rows,stats=self.build(current,past,future)
        self.assertEqual(rows[0]['chosen'][-1]['content'],'eligible future')
        self.assertEqual(rows[0]['rejected'][-1]['content'],'current bad')
        self.assertEqual(stats['retained_chosen_from_Mf'],1)

    def test_no_fallback_to_current_when_preferred_future_is_over_cap(self):
        current=record(['short current','bad'],[5,0]);past=record(['past'],[0])
        rows,stats=self.build(current,past,record(['x'*2501],[5]))
        self.assertEqual(rows,[])
        self.assertEqual(stats['filtered_no_chosen_within_character_limit'],1)

    def test_future_training_also_drops_equal_score_pairs(self):
        same=record(['first','second'],[5,5])
        rows,stats=self.build(same,same)
        self.assertEqual(rows,[])
        self.assertEqual(stats['filtered_equal_score_pairs'],1)

    def test_distinct_text_selection_uses_next_candidate_without_score_fallback(self):
        current=record(['same','different good',' same '],[5,5,0])
        rows,_=self.build(current,record(['past'],[1]))
        self.assertEqual(rows[0]['chosen'][-1]['content'],'different good')
        self.assertEqual(rows[0]['rejected'][-1]['content'],' same ')

    def test_invalid_extraction_removed_before_extrema_but_real_zero_is_valid(self):
        current=record(['invalid high','valid good','valid bad'],[5,4,0])
        current['judgments'][0]=['No score in this judgment']
        rows,stats=self.build(current,record(['past'],[1]))
        self.assertEqual(rows[0]['chosen'][-1]['content'],'valid good')
        self.assertEqual(rows[0]['rejected'][-1]['content'],'valid bad')
        self.assertEqual(stats['invalid_score_filter']['invalid_candidates_removed'],1)

    def test_cap_is_inclusive_unicode_characters(self):
        current=record(['字'*2500,'bad'],[5,0])
        rows,stats=self.build(current,current)
        self.assertEqual(len(rows[0]['chosen'][-1]['content']),2500)
        self.assertEqual(stats['chosen_over_character_limit_pairs'],0)


class InitializationTests(unittest.TestCase):
    def test_both_mf1_and_m2_start_at_m1_and_all_batches_are_32(self):
        with tempfile.TemporaryDirectory() as tmp:
            for filename,stage in [('config_full_iter0_tmp','Mf0'),('config_full_iter0','M1'),
                                   ('config_full_iter1_tmp','Mf1'),('config_full_iter1','M2')]:
                template=ROOT/f'settings/tsr/config/dpo/{filename}.yaml'
                config=yaml.safe_load(configure('dpo',template,Path(tmp),'/base',8).read_text())
                parent='M1' if stage in ('Mf1','M2') else 'M0'
                self.assertEqual(Path(config['model_name_or_path']).name,NAMES[parent])
                self.assertEqual(config['per_device_train_batch_size']*config['gradient_accumulation_steps']*8,32)
                self.assertEqual(config['save_strategy'],'no')
                self.assertFalse(config['do_eval'])

    def test_wrong_m2_future_initialization_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);template=root/'config_full_iter1.yaml'
            original=yaml.safe_load((ROOT/'settings/tsr/config/dpo/config_full_iter1.yaml').read_text())
            original['model_name_or_path']='../models/'+NAMES['Mf1']
            template.write_text(yaml.safe_dump(original))
            with self.assertRaisesRegex(ValueError,'must initialize from M1'):
                configure('dpo',template,root/'run','/base',8)


class DatasetFreezeTests(unittest.TestCase):
    def test_ignored_original_data_is_hashed_and_included_in_frozen_bundle(self):
        import hashlib
        from prepare_best_run import bundle_files
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir()
            path=root/'data/questions.jsonl';path.write_text('original data')
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'data/manifest.json').write_text(json.dumps({'files':{'data/questions.jsonl':{'sha256':checksum}}}))
            self.assertIn('data/questions.jsonl',bundle_files(root,['scripts/native.py','data/manifest.json']))
            path.write_text('changed data')
            with self.assertRaisesRegex(ValueError,'checksum'):
                bundle_files(root,['data/manifest.json'])


class TensorBoardTests(unittest.TestCase):
    def test_local_event_write_and_closed_file_sync(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from sync_training_metrics import sync
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'shared/run';local=root/'node-local'
            with patch.dict('os.environ',{'TSR_TENSORBOARD_LOCAL_ROOT':str(local)}):
                config=configure('sft',ROOT/'settings/sft/config/config.yaml',run,'/base',8)
                directory=Path(yaml.safe_load(config.read_text())['logging_dir'])
                self.assertTrue(directory.is_relative_to(local))
                self.assertFalse(directory.is_relative_to(run))
                with SummaryWriter(str(directory)) as writer:
                    writer.add_scalar('train/loss',1.25,5)
                destination=sync(config,run)
                events=EventAccumulator(str(destination));events.Reload()
                scalar=events.Scalars('train/loss')[0]
                self.assertEqual((scalar.step,scalar.value),(5,1.25))


class NativeJudgingTests(unittest.TestCase):
    @staticmethod
    def module(filename='self_generate_vllm_all_iterJudge.py'):
        stub=types.ModuleType('vllm')
        stub.LLM=object
        stub.SamplingParams=lambda **kwargs:types.SimpleNamespace(**kwargs)
        spec=importlib.util.spec_from_file_location('native_judge',ROOT/'tsr'/filename)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'vllm':stub}):
            spec.loader.exec_module(module)
        return module

    def test_combined_sr_preserves_valid_rejected_text(self):
        module=self.module('self_generate_vllm_all.py')
        generator=module.SelfRewardGenerator.__new__(module.SelfRewardGenerator)
        rows=generator.process_all(['q'],[['good','zero','invalid']],[[['Score: 5'],['Score: 0'],['No score']]])
        self.assertEqual(rows[0]['chosen_response'],'good')
        self.assertEqual(rows[0]['rejected_response'],'zero')

    def test_variable_candidate_counts_and_judge_temperature_restore(self):
        module=self.module();generator=module.SelfRewardGenerator.__new__(module.SelfRewardGenerator)
        original=types.SimpleNamespace(temperature=1.0,top_p=1.0,max_tokens=1024,stop=['EOS'])
        generator.sampling_params=original
        generator.format_chat=lambda chat:repr(chat)
        seen=[]
        def generate(prompts):
            seen.append(generator.sampling_params.temperature)
            return ['Score: 4']*len(prompts)
        generator.generate_batch=generate
        with patch.dict('os.environ',{'TSR_SELF_JUDGE_TEMPERATURE':'0'}),contextlib.redirect_stdout(io.StringIO()):
            result=generator.generate_all_judgments(['q1','q2'],[['a','b'],['c']],num_judgments=1)
        self.assertEqual([len(group) for group in result],[2,1])
        self.assertEqual(seen,[0.0])
        self.assertIs(generator.sampling_params,original)
        rows=generator.process_all(['q'],[['bad extraction','real zero']],[[['Score: 9'],['Score: 0']]])
        self.assertEqual(rows[0]['scores'],[None,0.0])


if __name__=='__main__':
    unittest.main()
