from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests


class HTTPClient:
    """
    A simple HTTP client for making GET and POST requests.
    Handles common functionality like headers, base URLs, and error handling.
    """

    def __init__(
        self, base_url: str = "", default_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the HTTP client.

        Args:
            base_url: Base URL for all requests
            default_headers: Default headers to include in all requests
        """
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers or {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a GET request and return response JSON and status code."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            return {"status_code": response.status_code, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {
                "status_code": (
                    getattr(response, "status_code", None)
                    if "response" in locals()
                    else None
                ),
                "error": str(e),
            }

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a POST request and return response JSON and status code."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.post(
                url, data=data, json=json_data, headers=headers
            )
            response.raise_for_status()
            return {"status_code": response.status_code, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {
                "status_code": (
                    getattr(response, "status_code", None)
                    if "response" in locals()
                    else None
                ),
                "error": str(e),
            }

    def patch(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a PATCH request and return response JSON and status code."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.patch(
                url, params=params, json=json_data, headers=headers
            )
            response.raise_for_status()
            return {"status_code": response.status_code, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {
                "status_code": (
                    getattr(response, "status_code", None)
                    if "response" in locals()
                    else None
                ),
                "error": str(e),
            }

    def put(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a PUT request and return response JSON and status code."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.put(
                url, params=params, json=json_data, headers=headers
            )
            response.raise_for_status()
            return {"status_code": response.status_code, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {
                "status_code": (
                    getattr(response, "status_code", None)
                    if "response" in locals()
                    else None
                ),
                "error": str(e),
            }

    def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a DELETE request and return response JSON and status code."""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.delete(url, params=params, headers=headers)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "data": response.json() if response.content else {},
            }
        except requests.exceptions.RequestException as e:
            return {
                "status_code": (
                    getattr(response, "status_code", None)
                    if "response" in locals()
                    else None
                ),
                "error": str(e),
            }

    def set_auth_token(self, token: str) -> None:
        """Set authentication token in headers."""
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def clear_auth_token(self) -> None:
        """Remove authentication token from headers."""
        self.session.headers.pop("Authorization", None)
