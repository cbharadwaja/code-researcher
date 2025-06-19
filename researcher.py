import argparse

# openai is implicitly used by OpenAIEmbeddings, which is now in indexer.py
# os is used by the moved classes, not directly here anymore.
# dataclasses, typing.List were for the moved classes.
# langchain components are used by the moved classes.
# agents.function_tool was for the moved tools.

from agents import Agent, Runner # These are used by build_agent and main
from code_researcher_agent import (
    CodeIndexer,
    CodeResearchTools,
    IndexNotReadyError # Though not explicitly handled in main, it's good practice to import it if it's part of the public API
)


# Comment about moved classes can be removed now or kept for history, removing for cleanliness.


def build_agent(tool_provider: CodeResearchTools) -> Agent:
    instructions = (
        "You are Code Researcher, an autonomous agent that answers questions about"
        " large codebases. Use the search_code tool to retrieve relevant code\n"
        "snippets and read_file to inspect files in detail. Provide helpful,"
        " concise answers referencing file paths when possible."
    )
    return Agent(
        name="CodeResearcher",
        instructions=instructions,
        tools=[tool_provider.search_code, tool_provider.read_file],
    )


def main() -> None:
    """Main entry point for the Code Researcher Agent CLI."""
    parser = argparse.ArgumentParser(description="Code Research Agent")
    parser.add_argument("--codebase", required=True, help="Path to the codebase")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument(
        "--extensions",
        nargs="*",
        help="File extensions to index (e.g., .py .java .c)",
        default=None,
    )
    args = parser.parse_args()

    try:
        indexer = CodeIndexer(args.codebase, allowed_extensions=args.extensions)
        indexer.build()

        tool_provider = CodeResearchTools(indexer)
        agent = build_agent(tool_provider)
        result = Runner.run_sync(agent, args.question)
        print(result.final_output)
    except IndexNotReadyError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
