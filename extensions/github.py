"""
JARVIS Extensions - GitHub Integration
"""

import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class GitHubIntegration:
    """GitHub integration class for RAVANA automation agent."""
    
    def __init__(self, token: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "JARVIS-Mark5/1.0"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def search_repositories(self, query: str, language: str = None, sort: str = "stars") -> List[Dict[str, Any]]:
        """Search for repositories on GitHub."""
        try:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": sort,
                "per_page": 10
            }
            if language:
                params["q"] += f" language:{language}"
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("items", [])
            
        except Exception as e:
            self.logger.error(f"GitHub search error: {e}")
            return []
    
    def get_user_repositories(self, username: str) -> List[Dict[str, Any]]:
        """Get repositories for a specific user."""
        try:
            url = f"{self.base_url}/users/{username}/repos"
            params = {
                "sort": "updated",
                "per_page": 20
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"GitHub user repos error: {e}")
            return []
    
    def get_repository_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a repository."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"GitHub repo info error: {e}")
            return None
    
    def clone_repository(self, clone_url: str, target_dir: str) -> bool:
        """Clone a repository to local directory."""
        try:
            import subprocess
            import os
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            result = subprocess.run(
                ["git", "clone", clone_url, target_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully cloned repository to {target_dir}")
                return True
            else:
                self.logger.error(f"Failed to clone repository: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"GitHub clone error: {e}")
            return False
    
    def create_issue(self, owner: str, repo: str, title: str, body: str, labels: List[str] = None) -> Optional[Dict[str, Any]]:
        """Create an issue in a repository."""
        try:
            if not self.token:
                self.logger.error("GitHub token required to create issues")
                return None
            
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            data = {
                "title": title,
                "body": body
            }
            if labels:
                data["labels"] = labels
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"GitHub create issue error: {e}")
            return None
    
    def get_trending_repositories(self, language: str = None, since: str = "daily") -> List[Dict[str, Any]]:
        """Get trending repositories."""
        try:
            # Note: GitHub API doesn't have a direct trending endpoint
            # This is a simplified implementation
            query = "stars:>1000"
            if language:
                query += f" language:{language}"
            
            return self.search_repositories(query, sort="stars")
            
        except Exception as e:
            self.logger.error(f"GitHub trending error: {e}")
            return []

class GitHubExtension:
    """GitHub integration extension for JARVIS."""
    
    def __init__(self, token: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "JARVIS-Mark5/1.0"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def search_repositories(self, query: str, language: str = None, sort: str = "stars") -> List[Dict[str, Any]]:
        """Search for repositories on GitHub."""
        try:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": sort,
                "per_page": 10
            }
            if language:
                params["q"] += f" language:{language}"
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("items", [])
            
        except Exception as e:
            self.logger.error(f"GitHub search error: {e}")
            return []
    
    def get_user_repositories(self, username: str) -> List[Dict[str, Any]]:
        """Get repositories for a specific user."""
        try:
            url = f"{self.base_url}/users/{username}/repos"
            params = {
                "sort": "updated",
                "per_page": 20
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"GitHub user repos error: {e}")
            return []
    
    def get_repository_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a repository."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"GitHub repo info error: {e}")
            return None
    
    def clone_repository(self, clone_url: str, target_dir: str) -> bool:
        """Clone a repository to local directory."""
        try:
            import subprocess
            import os
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            result = subprocess.run(
                ["git", "clone", clone_url, target_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully cloned repository to {target_dir}")
                return True
            else:
                self.logger.error(f"Failed to clone repository: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"GitHub clone error: {e}")
            return False
    
    def create_issue(self, owner: str, repo: str, title: str, body: str, labels: List[str] = None) -> Optional[Dict[str, Any]]:
        """Create an issue in a repository."""
        try:
            if not self.token:
                self.logger.error("GitHub token required to create issues")
                return None
            
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            data = {
                "title": title,
                "body": body
            }
            if labels:
                data["labels"] = labels
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"GitHub create issue error: {e}")
            return None
    
    def get_trending_repositories(self, language: str = None, since: str = "daily") -> List[Dict[str, Any]]:
        """Get trending repositories."""
        try:
            # Note: GitHub API doesn't have a direct trending endpoint
            # This is a simplified implementation
            query = "stars:>1000"
            if language:
                query += f" language:{language}"
            
            return self.search_repositories(query, sort="stars")
            
        except Exception as e:
            self.logger.error(f"GitHub trending error: {e}")
            return []

def github_function(action: str, **kwargs) -> Dict[str, Any]:
    """Main GitHub function for JARVIS integration."""
    try:
        github = GitHubExtension(kwargs.get('token'))
        
        if action == "search_repos":
            return {
                "success": True,
                "data": github.search_repositories(
                    kwargs.get('query', ''),
                    kwargs.get('language'),
                    kwargs.get('sort', 'stars')
                )
            }
        elif action == "get_user_repos":
            return {
                "success": True,
                "data": github.get_user_repositories(kwargs.get('username', ''))
            }
        elif action == "get_repo_info":
            return {
                "success": True,
                "data": github.get_repository_info(
                    kwargs.get('owner', ''),
                    kwargs.get('repo', '')
                )
            }
        elif action == "clone_repo":
            return {
                "success": github.clone_repository(
                    kwargs.get('clone_url', ''),
                    kwargs.get('target_dir', './cloned_repo')
                ),
                "message": "Repository cloned successfully" if github.clone_repository(
                    kwargs.get('clone_url', ''),
                    kwargs.get('target_dir', './cloned_repo')
                ) else "Failed to clone repository"
            }
        elif action == "create_issue":
            return {
                "success": True,
                "data": github.create_issue(
                    kwargs.get('owner', ''),
                    kwargs.get('repo', ''),
                    kwargs.get('title', ''),
                    kwargs.get('body', ''),
                    kwargs.get('labels', [])
                )
            }
        elif action == "trending":
            return {
                "success": True,
                "data": github.get_trending_repositories(
                    kwargs.get('language'),
                    kwargs.get('since', 'daily')
                )
            }
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
            
    except Exception as e:
        logger.error(f"GitHub function error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function for testing."""
    # Test the GitHub extension
    github = GitHubExtension()
    
    # Search for Python repositories
    repos = github.search_repositories("python machine learning", language="python")
    print(f"Found {len(repos)} Python ML repositories")
    
    if repos:
        print(f"Top repo: {repos[0]['name']} - {repos[0]['stargazers_count']} stars")

if __name__ == "__main__":
    main()
