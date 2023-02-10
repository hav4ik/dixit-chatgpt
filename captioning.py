import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from transformers import (
    AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM,
    BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTFeatureExtractor
)


class BLIP2Wrapper:
    # ==================================================
    # Architectures                  Types
    # ==================================================
    # blip2_opt                      pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b
    # blip2_t5                       pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    # blip2                          pretrain, coco
    def __init__(self, name, model_type):
        # loads BLIP-2 pre-trained model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name, model_type=model_type,
            is_eval=True, device=device)

    def __call__(self, image, question=None,
                 max_length=72, num_beams=3, repetition_penalty=1.5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_ts = self.vis_processors["eval"](
            Image.fromarray(image)
        ).unsqueeze(0).to(device)

        if question is None:
            caption = self.model.generate(
                {
                    "image": image_ts
                },
                use_nucleus_sampling=False,
                max_length=max_length,
                min_length=32,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
            )
        else:
            caption = self.model.generate(
                {
                    "image": image_ts,
                    "prompt": f"Question: {question}? Answer:",
                },
                use_nucleus_sampling=False,
                max_length=max_length,
                min_length=32,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
            )
        return caption[0]


class CaptioningModel:
    def get_git_large_coco():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        git_processor_large = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        git_model_large = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
        git_model_large.to(device)
        return {
            'name': 'git_large',
            'model': git_model_large,
            'processor': git_processor_large,
            'tokenizer': None,
        }

    def get_git_base_coco():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        git_processor_large = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        git_model_base = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
        git_model_base.to(device)
        return {
            'name': 'git_base',
            'model': git_model_base,
            'processor': git_processor_large,
            'tokenizer': None,
        }

    def get_blip_base():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor_base = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model_base.to(device)
        return {
            'name': 'blip_base',
            'model': blip_model_base,
            'processor': blip_processor_base,
            'tokenizer': None,
        }

    def get_blip_large():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model_large.to(device)
        return {
            'name': 'blip_large',
            'model': blip_model_large,
            'processor': blip_processor_large,
            'tokenizer': None,
        }

    def get_vitgpt2():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vitgpt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        vitgpt_processor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        vitgpt_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        vitgpt_model.to(device)
        return {
            'name': 'vit_gpt2',
            'model': vitgpt_model,
            'processor': vitgpt_processor,
            'tokenizer': vitgpt_tokenizer,
        }

    def __init__(self, name="git_large_coco"):
        if name == 'git_large_coco':
            self.model = CaptioningModel.get_git_large_coco()
        elif name == 'git_base_coco':
            self.model = CaptioningModel.get_git_base_coco()
        elif name == 'blip_base':
            self.model = CaptioningModel.get_blip_base()
        elif name == 'blip_large':
            self.model = CaptioningModel.get_blip_large()
        elif name == 'vit_gpt2':
            self.model = CaptioningModel.get_vitgpt2()

    def __call__(self, image):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.model['processor'](images=image, return_tensors="pt").to(device)
        generated_ids = self.model['model'].generate(pixel_values=inputs.pixel_values, max_length=50)
        if self.model['tokenizer'] is not None:
            generated_caption = self.model['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            generated_caption = self.model['processor'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption


def generate_captions(image,
                      blip2_model=None,
                      git_large_coco_model=None,
                      blip_large_model=None,
                      ):
    captions, model_names = [], []

    # Blip2 model
    if blip2_model is not None:
        blip2_cap = blip2_model(image)
        captions.append(f"BLIP-V2: {blip2_cap.strip()}")
        model_names.append("BLIP-V2")
    
    # Git Large COCO model
    if git_large_coco_model is not None:
        git_large_coco_cap = git_large_coco_model(image)
        captions.append(f"Git-Large: {git_large_coco_cap.strip()}")
        model_names.append("Git-Large")

    # Blip Large model
    if blip_large_model is not None:
        blip_large_cap = blip_large_model(image)
        captions.append(f"BLIP-V1: {blip_large_cap.strip()}")
        model_names.append("BLIP-V1")

    return {"captions": '\n'.join(captions), "models": model_names}
