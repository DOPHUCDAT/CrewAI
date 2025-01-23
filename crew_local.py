from crewai import Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv

load_dotenv()
# --- Tools ---
pdf_search_tool = PDFSearchTool(
    pdf="./spice_wolf.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.2-vision")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    tools=[pdf_search_tool],
)

professional_writer_agent = Agent(
    role="Professional Writer",
    goal="Write professional blog based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise content based on the provided information.
        """
    ),
    tools=[],
)


# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the customer's questions based on the home inspection PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the home
        inspection PDF.

        Here is the customer's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the home inspection PDF.
        """,
    tools=[pdf_search_tool],
    agent=research_agent,
)

write_report_task = Task(
    description=(
        """
        Write a professional report to a contractor based on the research agent's findings.
        The report should clearly state the issues found in the specified section of the report
        and request a quote or action plan for guide customer.

        """
    ),
    expected_output="""
        Write a clear and concise report, creation of user-friendly layouts, seeking support to ensure accurate information and if you don't know
        just say that that it's out of your knowledge.
        """,
    tools=[],
    agent=professional_writer_agent,
)

# --- Crew ---
crew = Crew(
    tasks=[answer_customer_question_task, write_report_task],
    agents=[research_agent, professional_writer_agent],
    process=Process.sequential,
)

customer_question = input(
    "Enter your question: "
)
result = crew.kickoff(inputs={"customer_question": customer_question})
print(result)