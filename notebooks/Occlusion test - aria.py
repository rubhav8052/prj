# ============================================================
#  C O P Y R I G H T
# ------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# 
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ============================================================

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import os
import json


# In[2]:


# !pip install seaborn


# In[3]:


import torch
from transformers import AutoModelForCausalLM
import requests
import torch
from PIL import Image

from transformers import AriaProcessor, AriaForConditionalGeneration

# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images


# In[4]:


# from transformers import AutoProcessor
# from vllm import LLM, SamplingParams
# from PIL import Image


# In[5]:


import os
with open('/etc/environment', 'r') as f:
    envs = [v.split('=') for v in f.read().split('\n')]
    for v in envs:
        print(v[0])
        os.environ[v[0]] = v[1].replace('"', '')


# In[ ]:





# In[6]:


csv_paths = [os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/object_retrieval/AL_data/Vehicles_sampled_2.csv', 
             os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/object_retrieval/AL_data/Vehicles_sampled_3.csv']


# In[7]:


dfs = []
for path in csv_paths:
    dfs.append(pd.read_csv(path))
df = pd.concat(dfs)


# In[8]:


df.shape


# In[9]:


# df = df.drop(columns='index')


# In[10]:


# Histogram Plot
sns.histplot(data=df, x="occlusion", kde=False, bins=5)
plt.show()


# In[11]:


def show(im):
    from IPython.display import display, Image
    im = cv2.imencode('.png', im)[1]
    display(Image(im))


# In[12]:


print("occlusion is anything that covers parts of a vehicle. a occlusion is anything that is located between the vehicle and the camera (transparent objects are included). It is set to 0.0 if the object is fully visible. It is set to 20.0 if 1-20percent of object is occluded. It is set to 40.0 if 21-40percent of object is occluded. It is set to 60.0 if 41-60percent of object is occluded. It is set to 80.0 if 61-80percent of object is occluded. It is set to 99.0 if 81-99percent of object is occluded.")


# In[25]:


from IPython.display import display, Markdown
with open('prompts/occlusion prompt-deepseek.md', 'r') as f:
    prompt = f.read()
display(Markdown(prompt))


# In[ ]:





# In[14]:


model_id_or_path = "rhymes-ai/Aria"
model = AriaForConditionalGeneration.from_pretrained(
    model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16
)

processor = AriaProcessor.from_pretrained(model_id_or_path)


# In[15]:


def crop_with_context(image, bbox, context_percent):
   
    h, w, c = image.shape  # Get image dimensions

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Compute additional context in pixels
    x_expand = int((x_max - x_min) * context_percent)
    y_expand = int((y_max - y_min) * context_percent)

    # Compute new expanded bounding box
    new_x_min = x_min - x_expand
    new_y_min = y_min - y_expand
    new_x_max = x_max + x_expand
    new_y_max = y_max + y_expand

    # Compute necessary padding if out of bounds
    pad_top = max(0, -new_y_min)
    pad_bottom = max(0, new_y_max - h)
    pad_left = max(0, -new_x_min)
    pad_right = max(0, new_x_max - w)

    # Adjust bbox to remain within image boundaries
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(w, new_x_max)
    new_y_max = min(h, new_y_max)

    # Crop the image
    cropped_img = image[new_y_min:new_y_max, new_x_min:new_x_max]

    # Add black padding if needed
    cropped_img = cv2.copyMakeBorder(
        cropped_img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )

    return cropped_img


# In[16]:


def prepare_img(row):
    full_img_path = os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/object_retrieval/AL_data'+f'/vehicle_sampled/{row["img_sha"]}.png'
    im = cv2.imread(full_img_path)
    box = np.int0([row.x0, row.y0, row.x1, row.y1])
    im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
    processed_img = crop_with_context(im, box, 0.5)
    # show(im[box[1]:box[3], box[0]:box[2]])
    # show(processed_img)
    cv2.imwrite('vehicle_imgs/'+str(row['index'])+'.png', processed_img)


# In[17]:


ouput_path = 'aria_results_boxed_image_v3.json'


# In[ ]:


results = []
for idx, row in tqdm(df.sample(10).iterrows(), total=len(df)):
    prepare_img(row)
    img_path = 'vehicle_imgs/'+str(row['index'])+'.png'
    image = Image.open(img_path)

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"text": prompt, "type": "text"},
        ],
    }
]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    inputs.to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        do_sample=True,
        temperature=0.9,
    )
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(output_ids, skip_special_tokens=True)

    # for img_name, output in zip(batch_images, outputs):
    #     generated_text = output.outputs[0].text if output.outputs else ""
    #     result = {"image_name": os.path.basename(img_name), "output": generated_text}
    results.append(response)
    show(cv2.imread(img_path))
    print('GT:', row['occlusion'], response)
    # if idx%100==0:
    # with open(ouput_path, 'w') as f:
    #     f.write(json.dumps(results, indent=2))
    


# In[ ]:


# with open(ouput_path, 'w') as f:
#     f.write(json.dumps(results, indent=2))


# In[ ]:





# In[ ]:


# results


# In[ ]:




