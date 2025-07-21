import os
from typing import Type

import google.generativeai as genai
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from settings import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)


class DescribeImageInput(BaseModel):
    """Input schema for DescribeImageTool."""

    image_path: str = Field(..., description="Path to the image to describe.")


class DescribeImageTool(BaseTool):
    name: str = "describe_image"
    description: str = "This tool takes the local image file path and returns a detailed description using Vision model."
    args_schema: Type[BaseModel] = DescribeImageInput

    def _run(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return f"Image not found at: {image_path}"

        model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            response = model.generate_content(
                [
                    "Describe this image in details and extract all details.",
                    {
                        "mime_type": "image/jpeg",  # adjust if needed
                        "data": image_data,
                    },
                ]
            )

            return response.text
        except Exception as e:
            return f"Error processing image: {e}"
