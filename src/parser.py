import fitz
import re
from src.models import Course

class PDFParser:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

    def extract_text(self):

        document = fitz.open(self.pdf_path)

        text = ""

        for page in document:

            text += page.get_text()

        return text
    
    def split_into_lines(self, text):
            """
            Split extracted PDF text into cleaned lines.
            """

            lines = []

            for line in text.splitlines():

                line = line.strip()

                if line:
                    lines.append(line)

            return lines

    
    def find_course_code(self, lines, start_index):
        """
        Search downwards from the course header to locate the course code.
        """

        for j in range(start_index, min(start_index + 75, len(lines))):

            line = lines[j].strip()
            match = re.search(r"\b\d{2,3}[A-Z]+\d{4}\b", line)

            if match:
                return match.group(), j

        return "", -1

    def extract_raw_courses(self, lines):
        """
        Extract complete course documents using the CBCS anchor.
        """

        courses = []
        course_positions = []

        # Pass 1: Find all course start positions
        for i, line in enumerate(lines):

            if "Choice Based Credit System" in line:

                # Find the course title (first non-empty line above)
                title_index = i - 1

                while title_index >= 0 and lines[title_index].strip() == "":
                    title_index -= 1

                course_name = lines[title_index].strip()

                # Find course code by searching downward from the title
                course_code, _ = self.find_course_code(lines, i)

                course_positions.append({
                    "course_name": course_name,
                    "course_code": course_code,
                    "start": title_index
                })

        # Pass 2: Build Course objects
        for idx, current in enumerate(course_positions):

            start = current["start"]

            if idx < len(course_positions) - 1:
                end = course_positions[idx + 1]["start"]
            else:
                end = len(lines)

            course = Course()

            course.course_name = current["course_name"]
            course.course_code = current["course_code"]
            course.raw_text = "\n".join(lines[start:end])

            courses.append(course)

        return courses
        
    def clean_value(self, value):
        """
        Clean extracted text from PDF artifacts.
        """

        if not value:
            return ""

        value = value.replace("Credits", "")

        value = value.replace("Course Name", "")

        value = value.replace("Type", "")

        return value.strip()
    
    