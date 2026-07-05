#!/usr/bin/env python
# coding: utf-8

# # Batch processing with the Batch API
# 
# The new Batch API allows to **create async batch jobs for a lower price and with higher rate limits**.
# 
# Batches will be completed within 24h, but may be processed sooner depending on global usage. 
# 
# Ideal use cases for the Batch API include:
# 
# - Tagging, captioning, or enriching content on a marketplace or blog
# - Categorizing and suggesting answers for support tickets
# - Performing sentiment analysis on large datasets of customer feedback
# - Generating summaries or translations for collections of documents or articles
# 
# and much more!
# 
# This cookbook will walk you through how to use the Batch API with a couple of practical examples.
# 
# We will start with an example to categorize movies using `gpt-4o-mini`, and then cover how we can use the vision capabilities of this model to caption images.
# 
# Please note that multiple models are available through the Batch API, and that you can use the same parameters in your Batch API calls as with the Chat Completions endpoint.

# ## Setup

# In[ ]:


# Make sure you have the latest version of the SDK available to use the Batch API
get_ipython().run_line_magic('pip', 'install openai --upgrade')


# In[1]:


import json
from openai import OpenAI
import pandas as pd
from IPython.display import Image, display


# In[2]:


# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI()


# ## First example: Categorizing movies
# 
# In this example, we will use `gpt-4o-mini` to extract movie categories from a description of the movie. We will also extract a 1-sentence summary from this description. 
# 
# We will use [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) to extract categories as an array of strings and the 1-sentence summary in a structured format. 
# 
# For each movie, we want to get a result that looks like this:
# 
# ```
# {
#     categories: ['category1', 'category2', 'category3'],
#     summary: '1-sentence summary'
# }
# ```

# ### Loading data
# 
# We will use the IMDB top 1000 movies dataset for this example. 

# In[3]:


dataset_path = "data/imdb_top_1000.csv"

df = pd.read_csv(dataset_path)
df.head()


# ### Processing step 
# 
# Here, we will prepare our requests by first trying them out with the Chat Completions endpoint.
# 
# Once we're happy with the results, we can move on to creating the batch file.

# In[4]:


categorize_system_prompt = '''
Your goal is to extract movie categories from movie descriptions, as well as a 1-sentence summary for these movies.
You will be provided with a movie description, and you will output a json object containing the following information:

{
    categories: string[] // Array of categories based on the movie description,
    summary: string // 1-sentence summary of the movie based on the movie description
}

Categories refer to the genre or type of the movie, like "action", "romance", "comedy", etc. Keep category names simple and use only lower case letters.
Movies can have several categories, but try to keep it under 3-4. Only mention the categories that are the most obvious based on the description.
'''

def get_categories(description):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.1,
    # This is to enable JSON mode, making sure responses are valid json objects
    response_format={ 
        "type": "json_object"
    },
    messages=[
        {
            "role": "system",
            "content": categorize_system_prompt
        },
        {
            "role": "user",
            "content": description
        }
    ],
    )

    return response.choices[0].message.content


# In[5]:


# Testing on a few examples
for _, row in df[:5].iterrows():
    description = row['Overview']
    title = row['Series_Title']
    result = get_categories(description)
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")


# ### Creating the batch file
# 
# The batch file, in the `jsonl` format, should contain one line (json object) per request.
# Each request is defined as such:
# 
# ```
# {
#     "custom_id": <REQUEST_ID>,
#     "method": "POST",
#     "url": "/v1/chat/completions",
#     "body": {
#         "model": <MODEL>,
#         "messages": <MESSAGES>,
#         // other parameters
#     }
# }
# ```
# 
# Note: the request ID should be unique per batch. This is what you can use to match results to the initial input files, as requests will not be returned in the same order.

# In[6]:


# Creating an array of json tasks

tasks = []

for index, row in df.iterrows():
    
    description = row['Overview']
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        }
    }
    
    tasks.append(task)


# In[7]:


# Creating the file

file_name = "data/batch_tasks_movies.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')


# ### Uploading the file

# In[8]:


batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)


# In[9]:


print(batch_file)


# ### Creating the batch job

# In[10]:


batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)


# ### Checking batch status
# 
# Note: this can take up to 24h, but it will usually be completed faster.
# 
# You can continue checking until the status is 'completed'.

# In[ ]:


batch_job = client.batches.retrieve(batch_job.id)
print(batch_job)


# ### Retrieving results

# In[13]:


result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content


# In[14]:


result_file_name = "data/batch_job_results_movies.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)


# In[15]:


# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)


# ### Reading results
# Reminder: the results are not in the same order as in the input file.
# Make sure to check the custom_id to match the results against the input requests

# In[16]:


# Reading only the first results
for res in results[:5]:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    movie = df.iloc[int(index)]
    description = movie['Overview']
    title = movie['Series_Title']
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")


# ## Second example: Captioning images
# 
# In this example, we will use `gpt-4-turbo` to caption images of furniture items. 
# 
# We will use the vision capabilities of the model to analyze the images and generate the captions.

# ### Loading data
# 
# We will use the Amazon furniture dataset for this example.

# In[12]:


dataset_path = "data/amazon_furniture_dataset.csv"
df = pd.read_csv(dataset_path)
df.head()


# ### Processing step 
# 
# Again, we will first prepare our requests with the Chat Completions endpoint, and create the batch file afterwards.

# In[13]:


caption_system_prompt = '''
Your goal is to generate short, descriptive captions for images of items.
You will be provided with an item image and the name of that item and you will output a caption that captures the most important information about the item.
If there are multiple items depicted, refer to the name provided to understand which item you should describe.
Your generated caption should be short (1 sentence), and include only the most important information about the item.
The most important information could be: the type of item, the style (if mentioned), the material or color if especially relevant and/or any distinctive features.
Keep it short and to the point.
'''

def get_caption(img_url, title):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,
    messages=[
        {
            "role": "system",
            "content": caption_system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": title
                },
                # The content type should be "image_url" to use gpt-4-turbo's vision capabilities
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    }
                },
            ],
        }
    ]
    )

    return response.choices[0].message.content


# In[14]:


# Testing on a few images
for _, row in df[:5].iterrows():
    img_url = row['primary_image']
    caption = get_caption(img_url, row['title'])
    img = Image(url=img_url)
    display(img)
    print(f"CAPTION: {caption}\n\n")


# ### Creating the batch job
# 
# As with the first example, we will create an array of json tasks to generate a `jsonl` file and use it to create the batch job.

# In[16]:


# Creating an array of json tasks

tasks = []

for index, row in df.iterrows():
    
    title = row['title']
    img_url = row['primary_image']
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": caption_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": title
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url
                            }
                        },
                    ],
                }
            ]            
        }
    }
    
    tasks.append(task)


# In[17]:


# Creating the file

file_name = "data/batch_tasks_furniture.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')


# In[18]:


# Uploading the file 

batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)


# In[19]:


# Creating the job

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)


# In[ ]:


batch_job = client.batches.retrieve(batch_job.id)
print(batch_job)


# ### Getting results
# 
# As with the first example, we can retrieve results once the batch job is done.
# 
# Reminder: the results are not in the same order as in the input file.
# Make sure to check the custom_id to match the results against the input requests

# In[41]:


# Retrieving result file

result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content


# In[42]:


result_file_name = "data/batch_job_results_furniture.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)


# In[43]:


# Loading data from saved file

results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)


# In[44]:


# Reading only the first results
for res in results[:5]:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    item = df.iloc[int(index)]
    img_url = item['primary_image']
    img = Image(url=img_url)
    display(img)
    print(f"CAPTION: {result}\n\n")


# ## Wrapping up
# 
# In this cookbook, we have seen two examples of how to use the new Batch API, but keep in mind that the Batch API works the same way as the Chat Completions endpoint, supporting the same parameters and most of the recent models (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo...).
# 
# By using this API, you can significantly reduce costs, so we recommend switching every workload that can happen async to a batch job with this new API.