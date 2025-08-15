from typing import Dict, Any, Optional, List
from .http_client import HTTPClient
import logging

from adaptiq.core.reporting import get_logger

get_logger()

class AdaptiqCloud:
    """
    Adaptiq API client.
    """
    
    def __init__(self):
        """
        Initialize the Adaptiq cloud client.
        """

        self.base_url = "https://api.getadaptiq.io"
        self.http_client = HTTPClient(base_url=self.base_url)
        self.status_key = "status_code"
        self.response_key = "data"
        self.error_key = "error"
        self.project_route = "projects"
        
    
    def send_run_results(self, data: Dict[str, Any]) -> bool:
        """
        Send run results to the Adaptiq projects endpoint.

        Args:
            data: The JSON payload containing run or project results

        Returns:
            bool: True if the request was successful (HTTP 201), False otherwise
        """

        try:

            
            response = self.http_client.post(endpoint= "/projects", json_data=data)
            
            # Check if request was successful
            if response.get(self.status_key) == 201:
                return True
            else:
                error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
                logging.error(f"Failed to send run results: {error_msg}")
                return False
                
        except Exception as e:
            logging.error(f"An error occurred while sending run results: {e}")
            return False

    
    def send_project_report(self, project_id: str, data: Dict[str, Any]) -> bool:
        """
        Send a project report to a specific project endpoint.
        
        Args:
            project_id: The project ID
            data: The report data to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = f"/{self.project_route}/{project_id}/reports"
        response = self.http_client.post(endpoint=endpoint, json_data=data)
        
        if response.get(self.status_key) in [200, 201]:
            return True
        else:
            error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
            logging.error(f"Failed to send project report: {error_msg}")
            return False
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project information by ID.
        
        Args:
            project_id: The project ID
            
        Returns:
            Dict containing project data if successful, None otherwise
        """
        endpoint = f"/{self.project_route}/{project_id}"
        response = self.http_client.get(endpoint=endpoint)
        
        if response.get(self.status_key) == 200:
            return response.get(self.response_key)
        else:
            error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
            logging.error(f"Failed to get project: {error_msg}")
            return None
    
    def list_projects(self, limit: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """
        List all projects.
        
        Args:
            limit: Optional limit on number of projects to return
            
        Returns:
            List of projects if successful, None otherwise
        """
        endpoint = f"/{self.project_route}"
        params = {"limit": limit} if limit else None

        response = self.http_client.get(endpoint= endpoint, params=params)
        
        if response.get(self.status_key) == 200:
            return response.get(self.response_key)
        else:
            error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
            logging.error(f"Failed to list projects: {error_msg}")
            return None
    
    def create_project(self, project_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new project.
        
        Args:
            project_data: Dictionary containing project information
            
        Returns:
            Created project data if successful, None otherwise
        """

        endpoint = f"/{self.project_route}"
        response = self.http_client.post(endpoint=endpoint, json_data=project_data)
        
        if response.get(self.status_key) in [200, 201]:
            return response.get(self.response_key)
        else:
            error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
            logging.error(f"Failed to create project: {error_msg}")
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
        endpoint = f"/{self.project_route}/{project_id}"
        response = self.http_client.patch(endpoint, json_data=project_data)
        
        if response.get(self.status_key) == 200:
            return response.get(self.response_key)
        else:
            error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
            logging.error(f"Failed to update project: {error_msg}")
            return None
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: The project ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = f"/{self.project_route}/{project_id}"
        response = self.http_client.delete(endpoint=endpoint)
        
        if response.get(self.status_key) in [200, 204]:
            return True
        else:
            error_msg = response.get(self.error_key, f"Request failed with status {response.get(self.status_key)}")
            logging.error(f"Failed to delete project: {error_msg}")
            return False