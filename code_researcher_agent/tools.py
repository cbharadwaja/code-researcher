import os

from agents import function_tool # Assuming agents package is in PYTHONPATH or a top-level package
# If agents is a local module at the same level as researcher_app, this import might need adjustment
# depending on how the whole project is structured and run.
# For now, direct import is assumed based on previous researcher.py structure.

from .indexer import CodeIndexer
# Note: IndexNotReadyError is not directly raised or handled by CodeResearchTools methods,
# but by the CodeIndexer it uses. So, no direct import of IndexNotReadyError here.


class CodeResearchTools:
    """Encapsulates code research tools that operate on an initialized CodeIndexer."""
    def __init__(self, indexer: CodeIndexer):
        """Initializes CodeResearchTools with a CodeIndexer instance."""
        self.indexer = indexer

    @function_tool
    def search_code(self, query: str) -> str:
        """Search the indexed code for relevant snippets."""
        return self.indexer.search(query)

    @function_tool
    def read_file(self, path: str) -> str:
        """Read a file from the codebase."""
        # Prevent path traversal:
        # 1. Normalize the path to resolve '..' components (e.g., dir/../file -> file).
        # 2. Get the real, absolute path to the target file and the base code directory.
        #    This resolves symbolic links and ensures canonical path representations.
        # 3. Check if the base directory is a common prefix of the target file's real path.
        #    If `os.path.commonpath([real_file_path, real_base_dir])` is not equal to `real_base_dir`,
        #    it means `real_file_path` is outside `real_base_dir`.
        full_path = os.path.normpath(os.path.join(self.indexer.code_dir, path))
        base_dir = os.path.realpath(self.indexer.code_dir)
        real_full_path = os.path.realpath(full_path)
        if not os.path.commonpath([real_full_path, base_dir]) == base_dir:
            raise ValueError("Invalid path: Path traversal attempt detected.")
        if not os.path.isfile(real_full_path):
            raise FileNotFoundError(path)
        with open(real_full_path, "r", encoding="utf-8") as f:
            return f.read()
