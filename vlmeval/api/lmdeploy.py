from functools import partial
import copy
import io
import requests
import yaml

from vlmeval.api.base import BaseAPI
from vlmeval.dataset import (DATASET_MODALITY, DATASET_TYPE, build_dataset,
                             infer_dataset_basename)
from vlmeval.smp import *
from vlmeval.vlm.internvl.utils import (build_mcq_cot_prompt, build_mpo_prompt,
                                        build_multi_choice_prompt,
                                        build_qa_cot_prompt, format_nav_prompt,
                                        pile_action_history)

from ..dataset import DATASET_MODALITY, DATASET_TYPE

logger = get_logger(__name__)

# load all the gui templates
upper_path = Path(__file__).parent.parent
with open(os.path.join(upper_path, "vlm/internvl/gui_template.yaml"), "r") as f:
    GUI_TEMPLATE = yaml.load(f, Loader=yaml.FullLoader)


class ModelAdapter:
    '''应对不同模型针对输入的特殊处理'''

    def dump_image(self, line, dataset):
        '''针对不同数据集的图像处理逻辑'''
        return self.dump_image_func(line)

    def override_model_args(self, dataset) -> dict:
        '''针对不同数据集指定不同的 system prompt'''
        return {}

    def use_custom_prompt(self, dataset: str, system_prompt: str | None = None) -> bool:
        '''判断指定数据集是否应当重载 prompt 构造逻辑'''
        return False

    def build_prompt(self, line, dataset: str | None = None):
        '''对样本构造 prompt 做特殊处理'''
        raise NotImplementedError

    def process_inputs(self, inputs: dict, dataset: str | None) -> dict:
        return inputs

    def process_payload(self, payload: dict, dataset: str | None = None) -> dict:
        '''对 API 请求的 payload 做特殊处理'''
        return payload
    
    def postprocess(self, response: str, dataset: str | None = None):
        return response

class InternVL2Adapter(ModelAdapter):

    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt
        self.cot_prompt = None
        self.screen_parse = False

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        assert dataset is not None
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset',
            'Physics', 'SFE', 'SFE-zh',
            'XLRS-Bench-lite', 'OmniEarth-Bench',
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if system_prompt is not None and '<think>' in system_prompt and listinstr(['MicroVQA', 'MSEarthMCQ', 'MMSci_DEV_MCQ', 'MMMU', 'VisuLogic'], dataset):
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        # if dataset is not ChartMimic, dump image (assert "image_path" in line)
        if not listinstr(['ChartMimic'], dataset):
            tgt_path = self.dump_image(line, dataset)
        else:
            input_figure_path_rel = line["input_figure"]
            ROOT = LMUDataRoot()
            img_root = os.path.join(ROOT, 'images', 'ChartMimic')
            input_figure_path = os.path.join(img_root, input_figure_path_rel)
            tgt_path = [input_figure_path]

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'GUI':
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config["placeholders"]:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)
        return message

    def get_max_num(self, dataset):
        if dataset is None:
            return 6
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        elif DATASET_TYPE(dataset) == 'GUI':
            return 12
        else:
            return 6

    def process_payload(self, payload, dataset):
        max_num = self.get_max_num(dataset)
        if max_num is not None:
            payload = copy.deepcopy(payload)
            messages = payload['messages']
            for msg in messages:
                if msg['type'] == 'image_url':
                    msg['image_url']['max_dynamic_patch'] = max_num
        return payload


class InternVL3Adapter(ModelAdapter):

    def __init__(self):
        self.cot_prompt = None
        self.screen_parse = True
        self.split_think = True

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def override_model_args(self, dataset):
        think_ds = [
            "MMMU", "MathVista", "SFE", "Physics", "MathVision",
            "OlympiadBench", "IPhO_2025", "MaCBench"
        ]
        no_think_ds = [
            "MMBench_V11", "MMStar", "RealWorldQA", "MicroVQA", "MSEarthMCQ",
            "XLRS-Bench-lite", "MMSci_DEV_MCQ"
        ]
        think_system = (
            "You are an expert reasoner with extensive experience in all "
            "areas. You approach problems through systematic thinking and "
            "rigorous reasoning. Your response should reflect deep "
            "understanding and precise logical thinking, making your "
            "solution path and reasoning clear to others. Please put your "
            "thinking process within <think>...</think> tags.")
        if listinstr(think_ds, dataset):
            return dict(system_prompt=think_system)
        else:
            # do_sample=False for non-thinking dataset
            return dict(temperature=0.)

    def use_custom_prompt(self, dataset, system_prompt=None):
        if dataset in [
                'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
                'optics_dataset', 'quantum_dataset', 'statistics_dataset',
                'Physics', 'SFE', 'SFE-zh', 'IPhO_2025', 'XLRS-Bench-lite',
                'OmniEarth-Bench'
        ]:
            return False
        elif listinstr([
                'MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT',
                'MMAlignBench', 'ScreenSpot', 'ChartQAPro', 'MMMU'
        ], dataset):
            return False
        elif DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        elif DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA', 'GUI']:
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)

        # if dataset is not ChartMimic, dump image (assert "image_path" in line)
        if listinstr(['ChartMimic'], dataset):
            input_figure_path_rel = line["input_figure"]
            ROOT = LMUDataRoot()
            img_root = os.path.join(ROOT, 'images', 'ChartMimic')
            input_figure_path = os.path.join(img_root, input_figure_path_rel)
            tgt_path = [input_figure_path]
        else:
            tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif DATASET_TYPE(dataset) == 'GUI':
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config["placeholders"]:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))

        return message

    def process_inputs(self, inputs: dict, dataset: str | None) -> dict:
        from vlmeval.vlm.internvl.utils import reorganize_prompt, build_video_prompt

        image_items = [x.copy() for x in inputs if x['type'] == 'image']
        image_num = len(image_items)
        prompt = reorganize_prompt(inputs, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        # Strange process only for MMMU
        if listinstr(['MMMU'], dataset) and len(image_items) > 0:
            image = Image.open(image_items[0]['value'])
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
            tmp_image = io.BytesIO()
            image.save(tmp_image, format='PNG')
            image_items[0]['value'] = tmp_image

        prompt = prompt.replace('<image>', '<IMAGE_TOKEN>')
        inputs = [*image_items, dict(type='text', value=prompt)]
        return inputs

    def get_max_num(self, dataset):
        if dataset is None:
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) =='VIDEO':
            return 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        elif DATASET_TYPE(dataset) == 'GUI':
            return 12
        else:
            return None

    def process_payload(self, payload, dataset):
        max_num = self.get_max_num(dataset)
        if max_num is not None:
            payload = copy.deepcopy(payload)
            for msg in payload['messages']:
                content = msg['content']
                if isinstance(content, dict) and content.get('type') == 'image_url':
                    content['image_url']['max_dynamic_patch'] = max_num
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            item['image_url']['max_dynamic_patch'] = max_num
        return payload

    def postprocess(self, response: str, dataset: str | None = None):
        if self.split_think and '<think>' in response and '</think>' in response:
            thinking, _, answer = response.partition('<think>')[-1].partition('</think>')
            return answer
        else:
            return response


class InternS1_1NoThinkAdapter(ModelAdapter):

    def __init__(self):
        self.cot_prompt = None
        self.screen_parse = True
        self.split_think = True

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        if dataset in [
                'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
                'optics_dataset', 'quantum_dataset', 'statistics_dataset',
                'Physics', 'SFE', 'SFE-zh', 'IPhO_2025', 'XLRS-Bench-lite',
                'OmniEarth-Bench'
        ]:
            return False
        elif listinstr([
                'MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT',
                'MMAlignBench', 'ScreenSpot', 'ChartQAPro', 'MMMU'
        ], dataset):
            return False
        elif DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        elif DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA', 'GUI']:
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)

        # if dataset is not ChartMimic, dump image (assert "image_path" in line)
        if listinstr(['ChartMimic'], dataset):
            input_figure_path_rel = line["input_figure"]
            ROOT = LMUDataRoot()
            img_root = os.path.join(ROOT, 'images', 'ChartMimic')
            input_figure_path = os.path.join(img_root, input_figure_path_rel)
            tgt_path = [input_figure_path]
        else:
            tgt_path = self.dump_image(line, dataset)

        if DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif DATASET_TYPE(dataset) == 'GUI':
            ds_basename = infer_dataset_basename(dataset)
            ds = build_dataset(dataset, skeleton=True)
            action_space = ds.get_action_space()
            traj_dict = ds.get_trajectory(line)

            prompt_config = GUI_TEMPLATE[ds_basename]
            if 'history' in prompt_config["placeholders"]:
                traj_dict['history'] = pile_action_history(traj_dict['history'])
            prompt = format_nav_prompt(
                (
                    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{task}</ref>"  # noqa: E501
                    if self.screen_parse
                    else prompt_config["template"]
                ),
                prompt_config["placeholders"],
                action_space=action_space,
                **traj_dict,
            )
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))

        return message

    def process_inputs(self, inputs: dict, dataset: str | None) -> dict:
        from vlmeval.vlm.internvl.utils import reorganize_prompt, build_video_prompt

        image_items = [x.copy() for x in inputs if x['type'] == 'image']
        image_num = len(image_items)
        prompt = reorganize_prompt(inputs, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        # Strange process only for MMMU
        if listinstr(['MMMU'], dataset) and len(image_items) > 0:
            image = Image.open(image_items[0]['value'])
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
            tmp_image = io.BytesIO()
            image.save(tmp_image, format='PNG')
            image_items[0]['value'] = tmp_image

        prompt = prompt.replace('<image>', '<IMAGE_TOKEN>')
        inputs = [*image_items, dict(type='text', value=prompt)]
        return inputs

    def postprocess(self, response: str, dataset: str | None = None):
        if self.split_think and '<think>' in response and '</think>' in response:
            thinking, _, answer = response.partition('<think>')[-1].partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        elif self.split_think and '</think>' in response:
            thinking, _, answer = response.partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        else:
            return response


class InternS1_1ThinkAdapter(ModelAdapter):

    def __init__(self):
        self.cot_prompt = None
        self.screen_parse = True
        self.split_think = True

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        if dataset in ['SFE', 'SFE-zh', 'IPhO_2025']:
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        if listinstr(['SFE'], dataset):
            return self.build_sfe_prompt(line, dataset)
        elif listinstr(['IPhO_2025'], dataset):
            return self.build_hipho_prompt(line, dataset)
        else:
            assert False, "custom prompt not supported for dataset: " + dataset

    def build_sfe_prompt(self, line, dataset):
        MCQ_PROMPT = (
            "You are an expert in {discipline} and need to solve the following question."
        )

        EXACT_MATCH_PROMPT = (
            "You are an expert in {discipline} and need to solve the following question."
        )

        OPEN_QUESTION_PROMPT = (
            "You are an expert in {discipline} and need to solve the following question."
        )
        tgt_path = self.dump_image(line, dataset)

        question_type = line['question_type']
        field = line['category']
        question = line['question']

        if question_type == 'exact_match':
            prompt = EXACT_MATCH_PROMPT.format(discipline=field)
            question = prompt + " " + question
        elif question_type == 'mcq':
            prompt = MCQ_PROMPT.format(discipline=field)
            question = prompt + " " + question
            if not pd.isna(line['A']):
                question += '\nChoices are:\n'
                for ch in string.ascii_uppercase[:15]:
                    if not pd.isna(line[ch]):
                        question += f'{ch}. {line[ch]}\n'
                    else:
                        break
        elif question_type == 'open_ended':
            prompt = OPEN_QUESTION_PROMPT.format(discipline=field)
            question = prompt + " " + question

        prompt_segs = question.split('<image>')
        assert len(prompt_segs) == len(tgt_path) + 1
        msgs = []
        for i in range(len(tgt_path)):
            text = prompt_segs[i].strip()
            if text != '':
                msgs.append(dict(type='text', value=text))
            msgs.append(dict(type='image', value=tgt_path[i]))
        text = prompt_segs[-1].strip()
        if text != '':
            msgs.append(dict(type='text', value=text))
        return msgs

    def build_hipho_prompt(self, line, dataset):
        """Build physics competition prompt"""
        def safe_str(val):
            return "" if pd.isna(val) or val == '' else str(val)

        context = safe_str(line.get('context', ''))
        question = safe_str(line['question'])
        information = safe_str(line.get('information', ''))

        SYSTEM_PROMPTS_EN = (
            'Please answer the problem adhering to the following rules:\n'
            '1. Please use LaTeX format to represent the variables and formulas used in the solution process and results.\n'
            '2. Please put the final answer(s) in \\boxed{}, note that the unit of the answer should not be included in \\boxed{}.\n'
            '3. If the problem requires multiple answers, list them in order, each in a separate \\boxed{}.\n'
            'Problem: Information:{information}\n'
            'Context:{context}\n'
            'Question: {problem}')
        system_prompt = SYSTEM_PROMPTS_EN
        formatted_prompt = (system_prompt.replace(
            '{context}',
            context).replace('{problem}',
                             question).replace('{information}', information))

        msgs = []

        # Check for real image data (excluding placeholders)
        image_val = str(line.get('image', '')).strip()

        if image_val and not image_val.startswith('NO_IMAGE_PLACEHOLDER_'):
            tgt_path = self.dump_image(line, dataset)

            if tgt_path and tgt_path != ['']:
                if isinstance(tgt_path, list):
                    msgs.extend([dict(type='image', value=p) for p in tgt_path])
                else:
                    msgs.append(dict(type='image', value=tgt_path))

        msgs.append(dict(type='text', value=formatted_prompt))

        return msgs


    def process_inputs(self, inputs: dict, dataset: str | None) -> dict:
        from vlmeval.vlm.internvl.utils import reorganize_prompt, build_video_prompt

        image_items = [x.copy() for x in inputs if x['type'] == 'image']
        image_num = len(image_items)
        prompt = reorganize_prompt(inputs, image_num, dataset=dataset)

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        # Strange process only for MMMU
        if listinstr(['MMMU'], dataset) and len(image_items) > 0:
            image = Image.open(image_items[0]['value'])
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
            tmp_image = io.BytesIO()
            image.save(tmp_image, format='PNG')
            image_items[0]['value'] = tmp_image

        prompt = prompt.replace('<image>', '<IMAGE_TOKEN>')
        inputs = [*image_items, dict(type='text', value=prompt)]
        return inputs

    def postprocess(self, response: str, dataset: str | None = None):
        if self.split_think and '<think>' in response and '</think>' in response:
            thinking, _, answer = response.partition('<think>')[-1].partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        elif self.split_think and '</think>' in response:
            thinking, _, answer = response.partition('</think>')
            logger.info('-----------Thinking-----------\n'
                        f'{thinking}\n'
                        '------------------------------')
            return answer
        else:
            return response


class CogVLM2Adapter(ModelAdapter):

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset, system_prompt=None):
        assert dataset is not None
        if DATASET_TYPE(dataset) in 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message


class Qwen3Adapter(ModelAdapter):
    def __init__(self, max_pixels: int | None) -> None:
        super().__init__()
        self.max_pixels = max_pixels

    def process_payload(self, payload, dataset):
        if self.max_pixels:
            payload = payload.copy()
            payload["mm_processor_kwargs"] = {"max_pixels": self.max_pixels}
        return payload


class LMDeployWrapper(BaseAPI):

    is_api: bool = True

    adapter_map = {
        'cogvlm2': CogVLM2Adapter,
        'internvl2': InternVL2Adapter,
        'internvl2-mpo-cot': partial(InternVL2Adapter, use_mpo_prompt=True),
        'internvl3': InternVL3Adapter,
        'interns1_1_no_think': InternS1_1NoThinkAdapter,
        'interns1_1_think': InternS1_1ThinkAdapter,
        'qwen3': Qwen3Adapter,
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 wait: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 timeout: int = 120,
                 api_base: str = None,
                 system_prompt: str = None,
                 custom_prompt: str = None,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = api_base or os.environ.get('LMDEPLOY_API_BASE', None)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base

        kwargs = {"max_tokens": 16384, 'temperature': 0.0, **kwargs}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
        logger.info(f'Following kwargs received: {self.default_kwargs}, will use as generation config. ')

        if model is None:
            model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
            resp = requests.get(model_url)
            model_id_list = [str(data['id']) for data in resp.json()['data']]
            self.model = model if model in model_id_list else model_id_list[0]
            logger.info(f'Automatically select model from api base: `{self.model}`')
        else:
            self.model = model
        logger.info(f'lmdeploy evaluate model: {self.model}')

        self.adapter: ModelAdapter | None = None
        if custom_prompt is None:
            custom_prompt = self.set_prompt_pattern(self.model)
            if custom_prompt is not None:
                self.adapter = self.adapter_map[custom_prompt]()
                logger.info(f'Automatically select model adapter {custom_prompt}')
        else:
            self.adapter = self.adapter_map[custom_prompt]()
            logger.info(f'Using specified model adapter {custom_prompt}')

    def set_dump_image(self, dump_image_func):
        if self.adapter is not None:
            self.adapter.dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset) -> bool:
        if self.adapter is not None:
            return self.adapter.use_custom_prompt(dataset, self.system_prompt)
        return False

    def build_prompt(self, line, dataset=None):
        if self.adapter is not None:
            return self.adapter.build_prompt(line, dataset)
        raise NotImplementedError

    def set_prompt_pattern(self, model_name) -> str | None:
        if 'Phi-3.5-Vision'.lower() in model_name.lower():
            self.max_tokens = 1000
            self.temperature = 0.0
        if 'cogvlm2-llama3-chat-19B'.lower() in model_name.lower():
            self.max_tokens = 2048
            self.temperature = 0.0
            return 'cogvlm2'
        if 'internvl' in model_name.lower():
            if 'mpo' in model_name.lower():
                return 'internvl2-mpo-cot'
            else:
                return 'internvl2'
        if 'qvq'.lower() in model_name.lower():
            self.max_tokens = 4096
            self.temperature = 0.0

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            text_num=0
            img_flag=0
            for msg in inputs:
                if msg['type'] == 'text':
                    if text_num > 0:
                        for cont in content_list:
                            if cont['type'] == 'text':
                                if img_flag == 0:
                                    cont['text'] += msg["value"]
                                else:
                                    cont['text'] += '<IMAGE_TOKEN>' * img_flag + f'\n{msg["value"]}'
                                    img_flag=0
                                break
                    else:
                        content_list.append(dict(type='text', text=msg['value']))
                    text_num += 1
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    extra_args = msg.copy()
                    extra_args.pop('type')
                    extra_args.pop('value')
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
                    img_flag += 1
            if img_flag != 0 and text_num>1:
                for cont in content_list:
                    if cont['type'] == 'text':
                        cont['text'] += '<IMAGE_TOKEN>' * img_flag
                        break
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs, system_prompt: str | None):
        input_msgs = []
        if system_prompt is not None:
            input_msgs.append(dict(role='system', content=system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, dataset=None, **kwargs) -> str:
        if self.adapter is not None:
            model_args = self.adapter.override_model_args(dataset)
            system_prompt = model_args.pop('system_prompt', self.system_prompt)
            inputs = self.adapter.process_inputs(inputs, dataset)
            kwargs.update(model_args)
        else:
            system_prompt = self.system_prompt
        input_msgs = self.prepare_inputs(inputs, system_prompt=system_prompt)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key}'
        }
        payload = dict(model=self.model, messages=input_msgs, n=1, **kwargs)
        if self.adapter is not None:
            payload = self.adapter.process_payload(payload, dataset=dataset)
        
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            if self.adapter is not None:
                print('-----------answer before postprocess------------------')
                if 'enable_thinking' in payload:
                    print('enable_thinking:', payload['enable_thinking'])
                print(answer)
                print('------------------------------------------------------')
                answer = self.adapter.postprocess(answer, dataset=dataset)
                                      
            # for internvl2-8b-mpo-cot
            if getattr(self, 'use_mpo_prompt', False):
                from ..vlm.internvl.utils import mpo_post_processing
                answer = mpo_post_processing(answer, kwargs.get('dataset'))
        except:
            pass
        return ret_code, answer, response


class LMDeployAPI(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(LMDeployAPI, self).generate(message, dataset=dataset)
