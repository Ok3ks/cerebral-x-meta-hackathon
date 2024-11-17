### Area 1 problem

Develop an AI-powered plant operations assistant that provides guidance on power plant procedures, solar installation optimization, and clean energy system maintenance, drawing from technical documentation and best practices

### Approach 

- Gather specific data points, i.e using RAG with manually added technical installations guides and best practices. This is because the problem is tailored, it depends on location geography, technical expertise working on electrical instruments and systems.

- An Agentic workflow, which works with the content of the RAG

- Make it more customised, by filtering queries into three sections,

- Problem can framed as a multimodal one  - (Audio Input, to accept queries and present responses , Text - to retrieve relevant contexts, and prompt a large language model for responses to a user's query). Vision can also be incorporated using large version of LLama such LLama 11B.

### Attempts 

- I tried to finetune an embedding model to improve context retrieval right in prototype.ipynb but no improvements likely because of the small size of the dataset. 

- I also noticed that in long instructions, LLama 3.2-3B performed better when instructions indicated urgency using words like only, paramount and were repeated right before requesting a response.

- I attempted an Agentic RAG, using llama-index package to retrieve relevant context, and extract needed information from context. After which llama-3.2-3B creates a coherent response from this. 

### Future Work

- Already converted LLama 3.2-3B.gguf will be inserted on a mobile device, as this presents best user experience for this application.

- Another approach instead of using RAG will be use a large language model for example 405B with more and vast knowledge, and create a critique model in the workflow step to assess output of this model without working on creating new data sources. Although more computationally intensive, it requires less work in curating needed data. It's worth experimenting if this approach described performs better in retrieval

- Main point of work will be improving inference on device, by using smaller models, perhaps offline support for areas with low connectivity.