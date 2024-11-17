import os
import chromadb
from llama_index.llms.groq import Groq
from chromadb.config import Settings
from llama_index.core.tools import ToolSelection, ToolOutput
from utils import retrieve_similar

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Event,
    Context,
    step,
)

client_ = chromadb.PersistentClient(path="./data/embeds")


class FirstStep(Event):
    query: str


class SolarStep(Event):
    query: str


class PowerPlantStep(Event):
    query: str


class CleanEnergySystemStep(Event):
    query: str


class InstructionEvent(Event):
    instruction: str
    context: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput



class InformationRequired(Event):
    information: str


class MyWorkflow(Workflow):

    @step
    async def setup(self, ev: StartEvent) -> FirstStep:  

        return FirstStep(query=ev.query)

    @step
    async def my_step(self, ev: FirstStep, ctx: Context) -> InstructionEvent:
        """Assess query"""

        GROK_API_KEY = os.getenv("GROK_API_KEY")
        llm = Groq(model="llama-3.2-3b-preview", api_key=GROK_API_KEY,)
        await ctx.set("llm", llm)
        await ctx.set("orig_query", ev.query)

        prompt = f"""
                You are an AI-powered plant operations assistant that provides guidance on power plant procedures, solar installation optimization, 
                and clean energy system maintenance, drawing from technical documentation and best practices"""

        request =  """What information would you need to provide clear answers to this {ev.query}. 
    #                             Only respond with this 
                """
        await ctx.set("prompt", prompt)
        
        response = llm.complete(prompt+request)

        similar = retrieve_similar(ev.query, client_, 'constructionom')
        similar = " ".join(similar[0])

        return InstructionEvent(instruction=response.text, context=similar)


    # @step
    # async def solar_panel_step(self, ev: SolarStep, ctx: Context) -> InstructionEvent:
    #     """This step processes the response"""

    #     base_prompt = f"""
    #                             The focus is solar energy. You are a top consultant 
    #                             who is a master at his craft. 

                                
    #                             Give plausible examples to drive your points, as appropriate. Everything should be done in moderation
    #                             """     
    #     request =       """
    #                             What information would you need to provide clear answers to this {ev.query}. 
    #                             Only respond with this "
    #                     """
        
    #     llm = await ctx.get("llm")
    #     await ctx.set("base_prompt", base_prompt)
    #     response = llm.complete(base_prompt + request)

    #     similar = retrieve_similar(ev.query, client_, 'SIO')
    #     similar = " ".join(similar[0])
    #     print(similar)

    #     await ctx.set("instruction_event", response)
    #     return InstructionEvent(instruction=response.text, context=similar)


    # @step
    # async def second_step(self, ev: SecondStep, ctx: Context) -> InstructionEvent:
        
    #     """This step processes initial query provide instructions needed by llama to user."""

    #     base_prompt = """
    #                             The focus is Power plant procedures, 
    #                         """
    #     request =           """
    #                             What information would you need to provide clear instructions for this {ev.query}"
    #                         """
    #     llm = await ctx.get("llm")
    #     await ctx.set("base_prompt", base_prompt)

    #     response = llm.complete(base_prompt+request)
    #     await ctx.set("instruction_event", response)

    #     similar = retrieve_similar(ev.query, client_, 'PPP')
    #     similar = " ".join(similar)

    #     return InstructionEvent(instruction=response.text, context=similar)

    # @step
    # async def clean_energy_step(self, ev: CleanEnergySystemStep, ctx: Context) -> InstructionEvent:
    #     """This step processes initial query provide instructions needed by llama to user."""

    #     base_prompt = """
    #                             The focus is Clean Energy Systems, """
    #     request =     """
    #                             What information would you need to provide clear instructions for this {ev.query}"
    #                             Only output information required, nothing more.
    #                         """
    #     llm = await ctx.get("llm")
    #     await ctx.set("base_prompt", base_prompt)

    #     response = llm.complete(base_prompt + request)
    #     await ctx.set("instruction_event", response)

    #     similar = retrieve_similar(ev.query, client_, 'CESM')
    #     similar = " ".join(similar)

    #     return InstructionEvent(instruction=response.text, context=similar)


    @step 
    async def search_context(self, ev: InstructionEvent,
                             ctx: Context) -> InformationRequired:
        """Instead of asking users to input, search the context."""

        info_req = ev.instruction
        context = ev.context

        search_prompt = f""" 

                        Search for all the information required as stated here {info_req}
                        within this {context}, and output in a format easy for you- the assistant - to comprehend

                        If information is not present in materials, try to find clues which can help inform your judgement,
                        If you're not confident, do not guess as lives are stake.
                        """
        llm = await ctx.get("llm")
        response = llm.complete(search_prompt)
        return InformationRequired(information=response.text)

    @step
    async def convert_user_input_step(self, ev: InformationRequired, ctx: Context) -> StopEvent:
        """This step converts supplied user input to an output."""

        llm = await ctx.get("llm")

        orig_query = await ctx.get("orig_query")
        prompt = await ctx.get("prompt")

        info_required = ev.information

        # user_input = input("Please supply requested information in the following format INFORMATION: RESPONSE")
        final_prompt = f"""

                        Given this request:{orig_query}.

                        Here's your prompt {prompt} and the information required {info_required}. 
                        
                        Feel free to complement with your knowledge base. 
                        Now please answer this query, you can do this.

                        """
        response = llm.complete(final_prompt)

        return StopEvent(result=response)


async def main():
    query = input("How can I assist you")
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run(query=query)
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())