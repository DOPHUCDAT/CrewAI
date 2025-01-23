from crewai import Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv

load_dotenv()

pdf_search_tool = PDFSearchTool(
    pdf="./spice_wolf.pdf",
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
    goal="Write professional reports based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise reports based on the provided information.
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

write_email_task = Task(
    description=(
        """
        - Write a professional report to a contractor based 
            on the research agent's findings.
        - The report should clearly state the issues found in the specified section 
            of the report and request a quote or action plan for guide user.

        """
    ),
    expected_output="""
        Write a clear and concise email that can be sent to a contractor to address the 
        issues found in the home inspection report.
        """,
    tools=[],
    agent=professional_writer_agent,
)

# --- Crew ---
crew = Crew(
    agents=[research_agent, professional_writer_agent],
    tasks=[answer_customer_question_task, write_email_task],
    process=Process.sequential,
)

customer_question = input("Enter the question: ")
result = crew.kickoff(inputs={"customer_question": customer_question})
print(result)