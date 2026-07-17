from dataclasses import dataclass, field

@dataclass
class Course:

    course_code: str = ""

    course_name: str = ""

    raw_text: str = ""

    semester: int = 0

    credits: int = 0

    course_type: str = ""

    lecture: int = 0

    tutorial: int = 0

    practical: int = 0

    project: int = 0