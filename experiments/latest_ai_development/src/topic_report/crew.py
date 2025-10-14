from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class TopicReporting():
    """Topic Reporting crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            output_file='output/report_researcher.md'
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True,
            output_file='output/report_reporting_analyst.md'
        )

    @agent
    def quality_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_reviewer'], # type: ignore[index]
            verbose=True,
            output_file='output/report_quality_reviewer.md'
        )

    @agent
    def critique_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['critique_agent'], # type: ignore[index]
            verbose=True,
            output_file='output/report_critique_agent.md'
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def research_critique_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_critique_task'], # type: ignore[index]
            output_file='output/report_research_critique.md'
        )

    @task
    def research_refinement_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_refinement_task'], # type: ignore[index]
            output_file='output/refined_research.md'
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='output/report_reporting_task.md'
        )

    @task
    def report_critique_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_critique_task'], # type: ignore[index]
            output_file='output/report_critique.md'
        )

    @task
    def quality_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_check_task'], # type: ignore[index]
            output_file='output/report_quality_check.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
