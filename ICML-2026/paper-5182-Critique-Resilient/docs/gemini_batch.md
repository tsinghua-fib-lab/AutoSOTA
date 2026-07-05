<br />

The Gemini Batch API is designed to process large volumes of requests asynchronously at[50% of the standard cost](https://ai.google.dev/gemini-api/docs/pricing). The target turnaround time is 24 hours, but in majority of cases, it is much quicker.

Use Batch API for large-scale, non-urgent tasks such as data pre-processing or running evaluations where an immediate response is not required.

## Creating a batch job

You have two ways to submit your requests in Batch API:

- **[Inline requests](https://ai.google.dev/gemini-api/docs/batch-api#inline-requests):** A list of[`GenerateContentRequest`](https://ai.google.dev/api/batch-mode#GenerateContentRequest)objects directly included in your batch creation request. This is suitable for smaller batches that keep the total request size under 20MB. The**output** returned from the model is a list of`inlineResponse`objects.
- **[Input file](https://ai.google.dev/gemini-api/docs/batch-api#input-file):** A[JSON Lines (JSONL)](https://jsonlines.org/)file where each line contains a complete[`GenerateContentRequest`](https://ai.google.dev/api/batch-mode#GenerateContentRequest)object. This method is recommended for larger requests. The**output** returned from the model is a JSONL file where each line is either a`GenerateContentResponse`or a status object.

### Inline requests

For a small number of requests, you can directly embed the[`GenerateContentRequest`](https://ai.google.dev/api/batch-mode#GenerateContentRequest)objects within your[`BatchGenerateContentRequest`](https://ai.google.dev/api/batch-mode#request-body). The following example calls the[`BatchGenerateContent`](https://ai.google.dev/api/batch-mode#google.ai.generativelanguage.v1beta.BatchService.BatchGenerateContent)method with inline requests:  

### Python


    from google import genai
    from google.genai import types

    client = genai.Client()

    # A list of dictionaries, where each is a GenerateContentRequest
    inline_requests = [
        {
            'contents': [{
                'parts': [{'text': 'Tell me a one-sentence joke.'}],
                'role': 'user'
            }]
        },
        {
            'contents': [{
                'parts': [{'text': 'Why is the sky blue?'}],
                'role': 'user'
            }]
        }
    ]

    inline_batch_job = client.batches.create(
        model="models/gemini-2.5-flash",
        src=inline_requests,
        config={
            'display_name': "inlined-requests-job-1",
        },
    )

    print(f"Created batch job: {inline_batch_job.name}")

### JavaScript


    import {GoogleGenAI} from '@google/genai';

    const ai = new GoogleGenAI({});

    const inlinedRequests = [
        {
            contents: [{
                parts: [{text: 'Tell me a one-sentence joke.'}],
                role: 'user'
            }]
        },
        {
            contents: [{
                parts: [{'text': 'Why is the sky blue?'}],
                role: 'user'
            }]
        }
    ]

    const response = await ai.batches.create({
        model: 'gemini-2.5-flash',
        src: inlinedRequests,
        config: {
            displayName: 'inlined-requests-job-1',
        }
    });

    console.log(response);

### REST

    curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:batchGenerateContent \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -X POST \
    -H "Content-Type:application/json" \
    -d '{
        "batch": {
            "display_name": "my-batch-requests",
            "input_config": {
                "requests": {
                    "requests": [
                        {
                            "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}]},
                            "metadata": {
                                "key": "request-1"
                            }
                        },
                        {
                            "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}]},
                            "metadata": {
                                "key": "request-2"
                            }
                        }
                    ]
                }
            }
        }
    }'

### Input file

For larger sets of requests, prepare a JSON Lines (JSONL) file. Each line in this file must be a JSON object containing a user-defined key and a request object, where the request is a valid[`GenerateContentRequest`](https://ai.google.dev/api/batch-mode#GenerateContentRequest)object. The user-defined key is used in the response to indicate which output is the result of which request. For example, the request with the key defined as`request-1`will have its response annotated with the same key name.

This file is uploaded using the[File API](https://ai.google.dev/gemini-api/docs/files). The maximum allowed file size for an input file is 2GB.

The following is an example of a JSONL file. You can save it in a file named`my-batch-requests.json`:  

    {"key": "request-1", "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}], "generation_config": {"temperature": 0.7}}}
    {"key": "request-2", "request": {"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}}

Similarly to inline requests, you can specify other parameters like system instructions, tools or other configurations in each request JSON.

You can upload this file using the[File API](https://ai.google.dev/gemini-api/docs/files)as shown in the following example. If you are working with multimodal input, you can reference other uploaded files within your JSONL file.  

### Python


    import json
    from google import genai
    from google.genai import types

    client = genai.Client()

    # Create a sample JSONL file
    with open("my-batch-requests.jsonl", "w") as f:
        requests = [
            {"key": "request-1", "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}]}},
            {"key": "request-2", "request": {"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}}
        ]
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Upload the file to the File API
    uploaded_file = client.files.upload(
        file='my-batch-requests.jsonl',
        config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
    )

    print(f"Uploaded file: {uploaded_file.name}")

### JavaScript


    import {GoogleGenAI} from '@google/genai';
    import * as fs from "fs";
    import * as path from "path";
    import { fileURLToPath } from 'url';

    const ai = new GoogleGenAI({});
    const fileName = "my-batch-requests.jsonl";

    // Define the requests
    const requests = [
        { "key": "request-1", "request": { "contents": [{ "parts": [{ "text": "Describe the process of photosynthesis." }] }] } },
        { "key": "request-2", "request": { "contents": [{ "parts": [{ "text": "What are the main ingredients in a Margherita pizza?" }] }] } }
    ];

    // Construct the full path to file
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const filePath = path.join(__dirname, fileName); // __dirname is the directory of the current script

    async function writeBatchRequestsToFile(requests, filePath) {
        try {
            // Use a writable stream for efficiency, especially with larger files.
            const writeStream = fs.createWriteStream(filePath, { flags: 'w' });

            writeStream.on('error', (err) => {
                console.error(`Error writing to file ${filePath}:`, err);
            });

            for (const req of requests) {
                writeStream.write(JSON.stringify(req) + '\n');
            }

            writeStream.end();

            console.log(`Successfully wrote batch requests to ${filePath}`);

        } catch (error) {
            // This catch block is for errors that might occur before stream setup,
            // stream errors are handled by the 'error' event.
            console.error(`An unexpected error occurred:`, error);
        }
    }

    // Write to a file.
    writeBatchRequestsToFile(requests, filePath);

    // Upload the file to the File API.
    const uploadedFile = await ai.files.upload({file: 'my-batch-requests.jsonl', config: {
        mimeType: 'jsonl',
    }});
    console.log(uploadedFile.name);

### REST

    tmp_batch_input_file=batch_input.tmp
    echo -e '{"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}], "generationConfig": {"temperature": 0.7}}\n{"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}' > batch_input.tmp
    MIME_TYPE=$(file -b --mime-type "${tmp_batch_input_file}")
    NUM_BYTES=$(wc -c < "${tmp_batch_input_file}")
    DISPLAY_NAME=BatchInput

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
    -D "${tmp_header_file}" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H "X-Goog-Upload-Protocol: resumable" \
    -H "X-Goog-Upload-Command: start" \
    -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
    -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
    -H "Content-Type: application/jsonl" \
    -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
    -H "Content-Length: ${NUM_BYTES}" \
    -H "X-Goog-Upload-Offset: 0" \
    -H "X-Goog-Upload-Command: upload, finalize" \
    --data-binary "@${tmp_batch_input_file}" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)

The following example calls the[`BatchGenerateContent`](https://ai.google.dev/api/batch-mode#google.ai.generativelanguage.v1beta.BatchService.BatchGenerateContent)method with the input file uploaded using File API:  

### Python

    from google import genai

    # Assumes `uploaded_file` is the file object from the previous step
    client = genai.Client()
    file_batch_job = client.batches.create(
        model="gemini-2.5-flash",
        src=uploaded_file.name,
        config={
            'display_name': "file-upload-job-1",
        },
    )

    print(f"Created batch job: {file_batch_job.name}")

### JavaScript

    // Assumes `uploadedFile` is the file object from the previous step
    const fileBatchJob = await ai.batches.create({
        model: 'gemini-2.5-flash',
        src: uploadedFile.name,
        config: {
            displayName: 'file-upload-job-1',
        }
    });

    console.log(fileBatchJob);

### REST

    # Set the File ID taken from the upload response.
    BATCH_INPUT_FILE='files/123456'
    curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:batchGenerateContent \
    -X POST \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H "Content-Type:application/json" \
    -d "{
        'batch': {
            'display_name': 'my-batch-requests',
            'input_config': {
                'file_name': '${BATCH_INPUT_FILE}'
            }
        }
    }"

When you create a batch job, you will get a job name returned. Use this name for[monitoring](https://ai.google.dev/gemini-api/docs/batch-api#batch-job-status)the job status as well as[retrieving the results](https://ai.google.dev/gemini-api/docs/batch-api#retrieve-batch-results)once the job completes.

The following is an example output that contains a job name:  


    Created batch job from file: batches/123456789

### Batch embedding support

You can use the Batch API to interact with the[Embeddings model](https://ai.google.dev/gemini-api/docs/embeddings)for higher throughput. To create an embeddings batch job with either[inline requests](https://ai.google.dev/gemini-api/docs/batch-api#inline-requests)or[input files](https://ai.google.dev/gemini-api/docs/batch-api#input-file), use the`batches.create_embeddings`API and specify the embeddings model.  

### Python

    from google import genai

    client = genai.Client()

    # Creating an embeddings batch job with an input file request:
    file_job = client.batches.create_embeddings(
        model="gemini-embedding-001",
        src={'file_name': uploaded_batch_requests.name},
        config={'display_name': "Input embeddings batch"},
    )

    # Creating an embeddings batch job with an inline request:
    batch_job = client.batches.create_embeddings(
        model="gemini-embedding-001",
        # For a predefined list of requests `inlined_requests`
        src={'inlined_requests': inlined_requests},
        config={'display_name': "Inlined embeddings batch"},
    )

### JavaScript

    // Creating an embeddings batch job with an input file request:
    let fileJob;
    fileJob = await client.batches.createEmbeddings({
        model: 'gemini-embedding-001',
        src: {fileName: uploadedBatchRequests.name},
        config: {displayName: 'Input embeddings batch'},
    });
    console.log(`Created batch job: ${fileJob.name}`);

    // Creating an embeddings batch job with an inline request:
    let batchJob;
    batchJob = await client.batches.createEmbeddings({
        model: 'gemini-embedding-001',
        // For a predefined a list of requests `inlinedRequests`
        src: {inlinedRequests: inlinedRequests},
        config: {displayName: 'Inlined embeddings batch'},
    });
    console.log(`Created batch job: ${batchJob.name}`);

Read the Embeddings section in the[Batch API cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Batch_mode.ipynb)for more examples.

### Request configuration

You can include any request configurations you would use in a standard non-batch request. For example, you could specify the temperature, system instructions or even pass in other modalities. The following example shows an example inline request that contains a system instruction for one of the requests:  

### Python

    inline_requests_list = [
        {'contents': [{'parts': [{'text': 'Write a short poem about a cloud.'}]}]},
        {'contents': [{
            'parts': [{
                'text': 'Write a short poem about a cat.'
                }]
            }],
        'config': {
            'system_instruction': {'parts': [{'text': 'You are a cat. Your name is Neko.'}]}}
        }
    ]

### JavaScript

    inlineRequestsList = [
        {contents: [{parts: [{text: 'Write a short poem about a cloud.'}]}]},
        {contents: [{parts: [{text: 'Write a short poem about a cat.'}]}],
         config: {systemInstruction: {parts: [{text: 'You are a cat. Your name is Neko.'}]}}}
    ]

Similarly can specify tools to use for a request. The following example shows a request that enables the[Google Search tool](https://ai.google.dev/gemini-api/docs/google-search):  

### Python

    inlined_requests = [
    {'contents': [{'parts': [{'text': 'Who won the euro 1998?'}]}]},
    {'contents': [{'parts': [{'text': 'Who won the euro 2025?'}]}],
     'config':{'tools': [{'google_search': {}}]}}]

### JavaScript

    inlineRequestsList = [
        {contents: [{parts: [{text: 'Who won the euro 1998?'}]}]},
        {contents: [{parts: [{text: 'Who won the euro 2025?'}]}],
         config: {tools: [{googleSearch: {}}]}}
    ]

You can specify[structured output](https://ai.google.dev/gemini-api/docs/structured-output)as well. The following example shows how to specify for your batch requests.  

### Python

    import time
    from google import genai
    from pydantic import BaseModel, TypeAdapter

    class Recipe(BaseModel):
        recipe_name: str
        ingredients: list[str]

    client = genai.Client()

    # A list of dictionaries, where each is a GenerateContentRequest
    inline_requests = [
        {
            'contents': [{
                'parts': [{'text': 'List a few popular cookie recipes, and include the amounts of ingredients.'}],
                'role': 'user'
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': list[Recipe]
            }
        },
        {
            'contents': [{
                'parts': [{'text': 'List a few popular gluten free cookie recipes, and include the amounts of ingredients.'}],
                'role': 'user'
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': list[Recipe]
            }
        }
    ]

    inline_batch_job = client.batches.create(
        model="models/gemini-2.5-flash",
        src=inline_requests,
        config={
            'display_name': "structured-output-job-1"
        },
    )

    # wait for the job to finish
    job_name = inline_batch_job.name
    print(f"Polling status for job: {job_name}")

    while True:
        batch_job_inline = client.batches.get(name=job_name)
        if batch_job_inline.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'):
            break
        print(f"Job not finished. Current state: {batch_job_inline.state.name}. Waiting 30 seconds...")
        time.sleep(30)

    print(f"Job finished with state: {batch_job_inline.state.name}")

    # print the response
    for i, inline_response in enumerate(batch_job_inline.dest.inlined_responses, start=1):
        print(f"\n--- Response {i} ---")

        # Check for a successful response
        if inline_response.response:
            # The .text property is a shortcut to the generated text.
            print(inline_response.response.text)

### JavaScript


    import {GoogleGenAI, Type} from '@google/genai';

    const ai = new GoogleGenAI({});

    const inlinedRequests = [
        {
            contents: [{
                parts: [{text: 'List a few popular cookie recipes, and include the amounts of ingredients.'}],
                role: 'user'
            }],
            config: {
                responseMimeType: 'application/json',
                responseSchema: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                    'recipeName': {
                        type: Type.STRING,
                        description: 'Name of the recipe',
                        nullable: false,
                    },
                    'ingredients': {
                        type: Type.ARRAY,
                        items: {
                        type: Type.STRING,
                        description: 'Ingredients of the recipe',
                        nullable: false,
                        },
                    },
                    },
                    required: ['recipeName'],
                },
                },
            }
        },
        {
            contents: [{
                parts: [{text: 'List a few popular gluten free cookie recipes, and include the amounts of ingredients.'}],
                role: 'user'
            }],
            config: {
                responseMimeType: 'application/json',
                responseSchema: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                    'recipeName': {
                        type: Type.STRING,
                        description: 'Name of the recipe',
                        nullable: false,
                    },
                    'ingredients': {
                        type: Type.ARRAY,
                        items: {
                        type: Type.STRING,
                        description: 'Ingredients of the recipe',
                        nullable: false,
                        },
                    },
                    },
                    required: ['recipeName'],
                },
                },
            }
        }
    ]

    const inlinedBatchJob = await ai.batches.create({
        model: 'gemini-2.5-flash',
        src: inlinedRequests,
        config: {
            displayName: 'inlined-requests-job-1',
        }
    });

The following shows an example output of this job:  

    --- Response 1 ---
    [
      {
        "recipe_name": "Chocolate Chip Cookies",
        "ingredients": [
          "1 cup (2 sticks) unsalted butter, softened",
          "3/4 cup granulated sugar",
          "3/4 cup packed light brown sugar",
          "1 large egg",
          "1 teaspoon vanilla extract",
          "2 1/4 cups all-purpose flour",
          "1 teaspoon baking soda",
          "1/2 teaspoon salt",
          "1 1/2 cups chocolate chips"
        ]
      },
      {
        "recipe_name": "Oatmeal Raisin Cookies",
        "ingredients": [
          "1 cup (2 sticks) unsalted butter, softened",
          "1 cup packed light brown sugar",
          "1/2 cup granulated sugar",
          "2 large eggs",
          "1 teaspoon vanilla extract",
          "1 1/2 cups all-purpose flour",
          "1 teaspoon baking soda",
          "1 teaspoon ground cinnamon",
          "1/2 teaspoon salt",
          "3 cups old-fashioned rolled oats",
          "1 cup raisins"
        ]
      },
      {
        "recipe_name": "Sugar Cookies",
        "ingredients": [
          "1 cup (2 sticks) unsalted butter, softened",
          "1 1/2 cups granulated sugar",
          "1 large egg",
          "1 teaspoon vanilla extract",
          "2 3/4 cups all-purpose flour",
          "1 teaspoon baking powder",
          "1/2 teaspoon salt"
        ]
      }
    ]

    --- Response 2 ---
    [
      {
        "recipe_name": "Gluten-Free Chocolate Chip Cookies",
        "ingredients": [
          "1 cup (2 sticks) unsalted butter, softened",
          "3/4 cup granulated sugar",
          "3/4 cup packed light brown sugar",
          "2 large eggs",
          "1 teaspoon vanilla extract",
          "2 1/4 cups gluten-free all-purpose flour blend (with xanthan gum)",
          "1 teaspoon baking soda",
          "1/2 teaspoon salt",
          "1 1/2 cups chocolate chips"
        ]
      },
      {
        "recipe_name": "Gluten-Free Peanut Butter Cookies",
        "ingredients": [
          "1 cup (250g) creamy peanut butter",
          "1/2 cup (100g) granulated sugar",
          "1/2 cup (100g) packed light brown sugar",
          "1 large egg",
          "1 teaspoon vanilla extract",
          "1/2 teaspoon baking soda",
          "1/4 teaspoon salt"
        ]
      },
      {
        "recipe_name": "Gluten-Free Oatmeal Raisin Cookies",
        "ingredients": [
          "1/2 cup (1 stick) unsalted butter, softened",
          "1/2 cup granulated sugar",
          "1/2 cup packed light brown sugar",
          "1 large egg",
          "1 teaspoon vanilla extract",
          "1 cup gluten-free all-purpose flour blend",
          "1/2 teaspoon baking soda",
          "1/2 teaspoon ground cinnamon",
          "1/4 teaspoon salt",
          "1 1/2 cups gluten-free rolled oats",
          "1/2 cup raisins"
        ]
      }
    ]

## Monitoring job status

Use the operation name obtained when creating the batch job to poll its status. The state field of the batch job will indicate its current status. A batch job can be in one of the following states:

- `JOB_STATE_PENDING`: The job has been created and is waiting to be processed by the service.
- `JOB_STATE_RUNNING`: The job is in progress.
- `JOB_STATE_SUCCEEDED`: The job completed successfully. You can now retrieve the results.
- `JOB_STATE_FAILED`: The job failed. Check the error details for more information.
- `JOB_STATE_CANCELLED`: The job was cancelled by the user.
- `JOB_STATE_EXPIRED`: The job has expired because it was running or pending for more than 48 hours. The job will not have any results to retrieve. You can try submitting the job again or splitting up the requests into smaller batches.

You can poll the job status periodically to check for completion.  

### Python

    import time
    from google import genai

    client = genai.Client()

    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = "YOUR_BATCH_JOB_NAME"  # (e.g. 'batches/your-batch-id')
    batch_job = client.batches.get(name=job_name)

    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
      print(f"Current state: {batch_job.state.name}")
      time.sleep(30) # Wait for 30 seconds before polling again
      batch_job = client.batches.get(name=job_name)

    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")

### JavaScript

    // Use the name of the job you want to check
    // e.g., inlinedBatchJob.name from the previous step
    let batchJob;
    const completedStates = new Set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ]);

    try {
        batchJob = await ai.batches.get({name: inlinedBatchJob.name});
        while (!completedStates.has(batchJob.state)) {
            console.log(`Current state: ${batchJob.state}`);
            // Wait for 30 seconds before polling again
            await new Promise(resolve => setTimeout(resolve, 30000));
            batchJob = await client.batches.get({ name: batchJob.name });
        }
        console.log(`Job finished with state: ${batchJob.state}`);
        if (batchJob.state === 'JOB_STATE_FAILED') {
            // The exact structure of `error` might vary depending on the SDK
            // This assumes `error` is an object with a `message` property.
            console.error(`Error: ${batchJob.state}`);
        }
    } catch (error) {
        console.error(`An error occurred while polling job ${batchJob.name}:`, error);
    }

## Retrieving results

Once the job status indicates your batch job has succeeded, the results are available in the`response`field.  

### Python

    import json
    from google import genai

    client = genai.Client()

    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = "YOUR_BATCH_JOB_NAME"
    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            print(f"Results are in file: {result_file_name}")

            print("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            # Process file_content (bytes) as needed
            print(file_content.decode('utf-8'))

        # If batch job was created with inline request
        # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
        elif batch_job.dest and batch_job.dest.inlined_responses:
            # Results are inline
            print("Results are inline:")
            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                print(f"Response {i+1}:")
                if inline_response.response:
                    # Accessing response, structure may vary.
                    try:
                        print(inline_response.response.text)
                    except AttributeError:
                        print(inline_response.response) # Fallback
                elif inline_response.error:
                    print(f"Error: {inline_response.error}")
        else:
            print("No results found (neither file nor inline).")
    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")

### JavaScript

    // Use the name of the job you want to check
    // e.g., inlinedBatchJob.name from the previous step
    const jobName = "YOUR_BATCH_JOB_NAME";

    try {
        const batchJob = await ai.batches.get({ name: jobName });

        if (batchJob.state === 'JOB_STATE_SUCCEEDED') {
            console.log('Found completed batch:', batchJob.displayName);
            console.log(batchJob);

            // If batch job was created with a file destination
            if (batchJob.dest?.fileName) {
                const resultFileName = batchJob.dest.fileName;
                console.log(`Results are in file: ${resultFileName}`);

                console.log("Downloading result file content...");
                const fileContentBuffer = await ai.files.download({ file: resultFileName });

                // Process fileContentBuffer (Buffer) as needed
                console.log(fileContentBuffer.toString('utf-8'));
            }

            // If batch job was created with inline responses
            else if (batchJob.dest?.inlinedResponses) {
                console.log("Results are inline:");
                for (let i = 0; i < batchJob.dest.inlinedResponses.length; i++) {
                    const inlineResponse = batchJob.dest.inlinedResponses[i];
                    console.log(`Response ${i + 1}:`);
                    if (inlineResponse.response) {
                        // Accessing response, structure may vary.
                        if (inlineResponse.response.text !== undefined) {
                            console.log(inlineResponse.response.text);
                        } else {
                            console.log(inlineResponse.response); // Fallback
                        }
                    } else if (inlineResponse.error) {
                        console.error(`Error: ${inlineResponse.error}`);
                    }
                }
            }

            // If batch job was an embedding batch with inline responses
            else if (batchJob.dest?.inlinedEmbedContentResponses) {
                console.log("Embedding results found inline:");
                for (let i = 0; i < batchJob.dest.inlinedEmbedContentResponses.length; i++) {
                    const inlineResponse = batchJob.dest.inlinedEmbedContentResponses[i];
                    console.log(`Response ${i + 1}:`);
                    if (inlineResponse.response) {
                        console.log(inlineResponse.response);
                    } else if (inlineResponse.error) {
                        console.error(`Error: ${inlineResponse.error}`);
                    }
                }
            } else {
                console.log("No results found (neither file nor inline).");
            }
        } else {
            console.log(`Job did not succeed. Final state: ${batchJob.state}`);
            if (batchJob.error) {
                console.error(`Error: ${typeof batchJob.error === 'string' ? batchJob.error : batchJob.error.message || JSON.stringify(batchJob.error)}`);
            }
        }
    } catch (error) {
        console.error(`An error occurred while processing job ${jobName}:`, error);
    }

### REST

    BATCH_NAME="batches/123456" # Your batch job name

    curl https://generativelanguage.googleapis.com/v1beta/$BATCH_NAME \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H "Content-Type:application/json" 2> /dev/null > batch_status.json

    if jq -r '.done' batch_status.json | grep -q "false"; then
        echo "Batch has not finished processing"
    fi

    batch_state=$(jq -r '.metadata.state' batch_status.json)
    if [[ $batch_state = "JOB_STATE_SUCCEEDED" ]]; then
        if [[ $(jq '.response | has("inlinedResponses")' batch_status.json) = "true" ]]; then
            jq -r '.response.inlinedResponses' batch_status.json
            exit
        fi
        responses_file_name=$(jq -r '.response.responsesFile' batch_status.json)
        curl https://generativelanguage.googleapis.com/download/v1beta/$responses_file_name:download?alt=media \
        -H "x-goog-api-key: $GEMINI_API_KEY" 2> /dev/null
    elif [[ $batch_state = "JOB_STATE_FAILED" ]]; then
        jq '.error' batch_status.json
    elif [[ $batch_state == "JOB_STATE_CANCELLED" ]]; then
        echo "Batch was cancelled by the user"
    elif [[ $batch_state == "JOB_STATE_EXPIRED" ]]; then
        echo "Batch expired after 48 hours"
    fi

## Cancelling a batch job

You can cancel an ongoing batch job using its name. When a job is canceled, it stops processing new requests.  

### Python

    from google import genai

    client = genai.Client()

    # Cancel a batch job
    client.batches.cancel(name=batch_job_to_cancel.name)

### JavaScript

    await ai.batches.cancel({name: batchJobToCancel.name});

### REST

    BATCH_NAME="batches/123456" # Your batch job name

    # Cancel the batch
    curl https://generativelanguage.googleapis.com/v1beta/$BATCH_NAME:cancel \
    -H "x-goog-api-key: $GEMINI_API_KEY" \

    # Confirm that the status of the batch after cancellation is JOB_STATE_CANCELLED
    curl https://generativelanguage.googleapis.com/v1beta/$BATCH_NAME \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H "Content-Type:application/json" 2> /dev/null | jq -r '.metadata.state'

## Deleting a batch job

You can delete an existing batch job using its name. When a job is deleted, it stops processing new requests and is removed from the list of batch jobs.  

### Python

    from google import genai

    client = genai.Client()

    # Delete a batch job
    client.batches.delete(name=batch_job_to_delete.name)

### JavaScript

    await ai.batches.delete({name: batchJobToDelete.name});

### REST

    BATCH_NAME="batches/123456" # Your batch job name

    # Delete the batch job
    curl https://generativelanguage.googleapis.com/v1beta/$BATCH_NAME:delete \
    -H "x-goog-api-key: $GEMINI_API_KEY"

## Generating images in batch

If you're using[Gemini Nano Banana](https://ai.google.dev/gemini-api/docs/image-generation)and need to generate a lot of images, you can use the Batch API to get higher[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)in exchange for a turnaround of up to 24 hours.

You can either use inline requests for small batches of requests (under 20MB) or a JSONL input file for large batches (recommended for image generation):

Inline requestsInput file  

### Python

    import json
    import time
    import base64
    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client()

    # 1. Create and upload file
    file_name = "my-batch-image-requests.jsonl"
    with open(file_name, "w") as f:
        requests = [
            {"key": "request-1", "request": {"contents": [{"parts": [{"text": "A big letter A surrounded by animals starting with the A letter"}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}},
            {"key": "request-2", "request": {"contents": [{"parts": [{"text": "A big letter B surrounded by animals starting with the B letter"}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}}
        ]
        for req in requests:
            f.write(json.dumps(req) + "\n")

    uploaded_file = client.files.upload(
        file=file_name,
        config=types.UploadFileConfig(display_name='my-batch-image-requests', mime_type='jsonl')
    )
    print(f"Uploaded file: {uploaded_file.name}")

    # 2. Create batch job
    file_batch_job = client.batches.create(
        model="gemini-2.5-flash-image",
        src=uploaded_file.name,
        config={
            'display_name': "file-image-upload-job-1",
        },
    )
    print(f"Created batch job: {file_batch_job.name}")

    # 3. Monitor job status
    job_name = file_batch_job.name
    print(f"Polling status for job: {job_name}")

    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
      print(f"Current state: {batch_job.state.name}")
      time.sleep(10) # Wait for 10 seconds before polling again
      batch_job = client.batches.get(name=job_name)

    print(f"Job finished with state: {batch_job.state.name}")

    # 4. Retrieve results
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        result_file_name = batch_job.dest.file_name
        print(f"Results are in file: {result_file_name}")
        print("Downloading result file content...")
        file_content_bytes = client.files.download(file=result_file_name)
        file_content = file_content_bytes.decode('utf-8')
        # The result file is also a JSONL file. Parse and print each line.
        for line in file_content.splitlines():
          if line:
            parsed_response = json.loads(line)
            if 'response' in parsed_response and parsed_response['response']:
                for part in parsed_response['response']['candidates'][0]['content']['parts']:
                  if part.get('text'):
                    print(part['text'])
                  elif part.get('inlineData'):
                    print(f"Image mime type: {part['inlineData']['mimeType']}")
                    data = base64.b64decode(part['inlineData']['data'])
            elif 'error' in parsed_response:
                print(f"Error: {parsed_response['error']}")
    elif batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")

### JavaScript

    import {GoogleGenAI} from '@google/genai';
    import * as fs from "fs";
    import * as path from "path";
    import { fileURLToPath } from 'url';

    const ai = new GoogleGenAI({});

    async function run() {
        // 1. Create and upload file
        const fileName = "my-batch-image-requests.jsonl";
        const requests = [
            { "key": "request-1", "request": { "contents": [{ "parts": [{ "text": "A big letter A surrounded by animals starting with the A letter" }] }], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]} } },
            { "key": "request-2", "request": { "contents": [{ "parts": [{ "text": "A big letter B surrounded by animals starting with the B letter" }] }], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]} } }
        ];
        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        const filePath = path.join(__dirname, fileName);

        try {
            const writeStream = fs.createWriteStream(filePath, { flags: 'w' });
            for (const req of requests) {
                writeStream.write(JSON.stringify(req) + '\n');
            }
            writeStream.end();
            console.log(`Successfully wrote batch requests to ${filePath}`);
        } catch (error) {
            console.error(`An unexpected error occurred writing file:`, error);
            return;
        }

        const uploadedFile = await ai.files.upload({file: fileName, config: { mimeType: 'jsonl' }});
        console.log(`Uploaded file: ${uploadedFile.name}`);

        // 2. Create batch job
        const fileBatchJob = await ai.batches.create({
            model: 'gemini-2.5-flash-image',
            src: uploadedFile.name,
            config: {
                displayName: 'file-image-upload-job-1',
            }
        });
        console.log(fileBatchJob);

        // 3. Monitor job status
        let batchJob;
        const completedStates = new Set([
            'JOB_STATE_SUCCEEDED',
            'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED',
        ]);

        try {
            batchJob = await ai.batches.get({name: fileBatchJob.name});
            while (!completedStates.has(batchJob.state)) {
                console.log(`Current state: ${batchJob.state}`);
                // Wait for 10 seconds before polling again
                await new Promise(resolve => setTimeout(resolve, 10000));
                batchJob = await ai.batches.get({ name: batchJob.name });
            }
            console.log(`Job finished with state: ${batchJob.state}`);
        } catch (error) {
            console.error(`An error occurred while polling job ${fileBatchJob.name}:`, error);
            return;
        }

        // 4. Retrieve results
        if (batchJob.state === 'JOB_STATE_SUCCEEDED') {
            if (batchJob.dest?.fileName) {
                const resultFileName = batchJob.dest.fileName;
                console.log(`Results are in file: ${resultFileName}`);
                console.log("Downloading result file content...");
                const fileContentBuffer = await ai.files.download({ file: resultFileName });
                const fileContent = fileContentBuffer.toString('utf-8');
                for (const line of fileContent.split('\n')) {
                    if (line) {
                        const parsedResponse = JSON.parse(line);
                        if (parsedResponse.response) {
                            for (const part of parsedResponse.response.candidates[0].content.parts) {
                                if (part.text) {
                                    console.log(part.text);
                                } else if (part.inlineData) {
                                    console.log(`Image mime type: ${part.inlineData.mimeType}`);
                                }
                            }
                        } else if (parsedResponse.error) {
                            console.error(`Error: ${parsedResponse.error}`);
                        }
                    }
                }
            } else {
                console.log("No result file found.");
            }
        } else if (batchJob.state === 'JOB_STATE_FAILED') {
             console.error(`Error: ${typeof batchJob.error === 'string' ? batchJob.error : batchJob.error.message || JSON.stringify(batchJob.error)}`);
        }
    }
    run();

### REST

    # 1. Create and upload file
    echo '{"key": "request-1", "request": {"contents": [{"parts": [{"text": "A big letter A surrounded by animals starting with the A letter"}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}}' > my-batch-image-requests.jsonl
    echo '{"key": "request-2", "request": {"contents": [{"parts": [{"text": "A big letter B surrounded by animals starting with the B letter"}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}}' >> my-batch-image-requests.jsonl

    # Follow File API guide to upload: https://ai.google.dev/gemini-api/docs/files#upload_a_file
    # This example assumes you have uploaded the file and set BATCH_INPUT_FILE to its name (e.g., files/abcdef123)
    BATCH_INPUT_FILE="files/your-uploaded-file-name"

    # 2. Create batch job
    printf -v request_data '{
        "batch": {
            "display_name": "my-batch-file-image-requests",
            "input_config": { "file_name": "%s" }
        }
    }' "$BATCH_INPUT_FILE"
    curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:batchGenerateContent \
      -X POST \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H "Content-Type:application/json" \
      -d "$request_data" > created_batch.json

    BATCH_NAME=$(jq -r '.name' created_batch.json)
    echo "Created batch job: $BATCH_NAME"

    # 3. Poll job status until completion by repeating the following command:
    curl https://generativelanguage.googleapis.com/v1beta/$BATCH_NAME \
      -H "x-goog-api-key: $GEMINI_API_KEY" \
      -H "Content-Type:application/json" > batch_status.json

    echo "Current status:"
    jq '.' batch_status.json

    # 4. If state is JOB_STATE_SUCCEEDED, download results file
    batch_state=$(jq -r '.state' batch_status.json)
    if [[ $batch_state = "JOB_STATE_SUCCEEDED" ]]; then
        responses_file_name=$(jq -r '.dest.fileName' batch_status.json)
        echo "Job succeeded. Downloading results from $responses_file_name..."
        curl https://generativelanguage.googleapis.com/download/v1beta/$responses_file_name:download?alt=media \
          -H "x-goog-api-key: $GEMINI_API_KEY" > batch_results.jsonl
        echo "Results saved to batch_results.jsonl"
    fi

## Technical details

- **Supported models:** Batch API supports a range of Gemini models. Refer to the[Models page](https://ai.google.dev/gemini-api/docs/models)for each model's support of Batch API. The supported modalities for Batch API are the same as what's supported on the interactive (or non-batch) API.
- **Pricing:** Batch API usage is priced at 50% of the standard interactive API cost for the equivalent model. See the[pricing page](https://ai.google.dev/gemini-api/docs/pricing)for details. Refer to the[rate limits page](https://ai.google.dev/gemini-api/docs/rate-limits#batch-mode)for details on rate limits for this feature.
- **Service Level Objective (SLO):**Batch jobs are designed to complete within a 24-hour turnaround time. Many jobs may complete much faster depending on their size and current system load.
- **Caching:** [Context caching](https://ai.google.dev/gemini-api/docs/caching)is enabled for batch requests. If a request in your batch results in a cache hit, the cached tokens are priced the same as for non-batch API traffic.

## Best practices

- **Use input files for large requests:** For a large number of requests, always use the file input method for better manageability and to avoid hitting request size limits for the[`BatchGenerateContent`](https://ai.google.dev/api/batch-mode#google.ai.generativelanguage.v1beta.BatchService.BatchGenerateContent)call itself. Note that there's a the 2GB file size limit per input file.
- **Error handling:** Check the`batchStats`for`failedRequestCount`after a job completes. If using file output, parse each line to check if it's a`GenerateContentResponse`or a status object indicating an error for that specific request. See the[troubleshooting guide](https://ai.google.dev/gemini-api/docs/troubleshooting#error-codes)for a complete set of error codes.
- **Submit jobs once:**The creation of a batch job is not idempotent. If you send the same creation request twice, two separate batch jobs will be created.
- **Break up very large batches:**While the target turnaround time is 24 hours, actual processing time can vary based on system load and job size. For large jobs, consider breaking them into smaller batches if intermediate results are needed sooner.

## What's next

- Check out the[Batch API notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Batch_mode.ipynb)for more examples.
- The OpenAI compatibility layer supports Batch API. Read the examples on the[OpenAI Compatibility](https://ai.google.dev/gemini-api/docs/openai#batch)page.