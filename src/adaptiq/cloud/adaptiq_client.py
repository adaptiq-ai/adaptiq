from typing import Dict, Any, Optional, List
from .http_client import HTTPClient


class AdaptiqClient(HTTPClient):
    """
    Adaptiq API client that extends HTTPClient with Adaptiq-specific functionality.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.getadaptiq.io"):
        """
        Initialize the Adaptiq client.
        
        Args:
            api_key: Optional API key for authentication
            base_url: Base URL for Adaptiq API (defaults to production)
        """
        super().__init__(base_url=base_url)
        
        if api_key:
            self.set_auth_token(api_key)
    
    def send_run_results(self, data: Dict[str, Any], timeout: int = 30) -> bool:
        """
        Send run results to the Adaptiq projects endpoint.

        Args:
            data: The JSON payload containing run or project results
            timeout: Request timeout in seconds (default: 30)

        Returns:
            bool: True if the request was successful (HTTP 201), False otherwise
        """
        # Override the default timeout by updating session
        original_timeout = getattr(self.session, 'timeout', None)
        
        try:
            # Temporarily set timeout on the session
            self.session.timeout = timeout
            
            response = self.post("/projects", json_data=data)
            
            # Check if request was successful
            if response.get("status_code") == 201:
                return True
            else:
                error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
                print(f"Failed to send run results: {error_msg}")
                return False
                
        except Exception as e:
            print(f"An error occurred while sending run results: {e}")
            return False
        finally:
            # Restore original timeout
            if original_timeout is not None:
                self.session.timeout = original_timeout
            elif hasattr(self.session, 'timeout'):
                delattr(self.session, 'timeout')
    
    def send_project_report(self, project_id: str, data: Dict[str, Any]) -> bool:
        """
        Send a project report to a specific project endpoint.
        
        Args:
            project_id: The project ID
            data: The report data to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = f"/projects/{project_id}/reports"
        response = self.post(endpoint, json_data=data)
        
        if response.get("status_code") in [200, 201]:
            return True
        else:
            error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
            print(f"Failed to send project report: {error_msg}")
            return False
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project information by ID.
        
        Args:
            project_id: The project ID
            
        Returns:
            Dict containing project data if successful, None otherwise
        """
        endpoint = f"/projects/{project_id}"
        response = self.get(endpoint)
        
        if response.get("status_code") == 200:
            return response.get("data")
        else:
            error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
            print(f"Failed to get project: {error_msg}")
            return None
    
    def list_projects(self, limit: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """
        List all projects.
        
        Args:
            limit: Optional limit on number of projects to return
            
        Returns:
            List of projects if successful, None otherwise
        """
        params = {"limit": limit} if limit else None
        response = self.get("/projects", params=params)
        
        if response.get("status_code") == 200:
            return response.get("data")
        else:
            error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
            print(f"Failed to list projects: {error_msg}")
            return None
    
    def create_project(self, project_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new project.
        
        Args:
            project_data: Dictionary containing project information
            
        Returns:
            Created project data if successful, None otherwise
        """
        response = self.post("/projects", json_data=project_data)
        
        if response.get("status_code") in [200, 201]:
            return response.get("data")
        else:
            error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
            print(f"Failed to create project: {error_msg}")
            return None
    
    def update_project(self, project_id: str, project_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing project.
        
        Args:
            project_id: The project ID
            project_data: Dictionary containing updated project information
            
        Returns:
            Updated project data if successful, None otherwise
        """
        endpoint = f"/projects/{project_id}"
        response = self.patch(endpoint, json_data=project_data)
        
        if response.get("status_code") == 200:
            return response.get("data")
        else:
            error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
            print(f"Failed to update project: {error_msg}")
            return None
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: The project ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = f"/projects/{project_id}"
        response = self.delete(endpoint)
        
        if response.get("status_code") in [200, 204]:
            return True
        else:
            error_msg = response.get("error", f"Request failed with status {response.get('status_code')}")
            print(f"Failed to delete project: {error_msg}")
            return False