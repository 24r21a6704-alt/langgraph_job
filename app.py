import os
import tempfile
import requests
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pypdf import PdfReader


# ============================================================
# 1. GOOGLE GEMINI API KEY
# ============================================================

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Google Gemini API key not found. "
        "Please add GOOGLE_API_KEY in Render Environment Variables."
    )


# ============================================================
# 2. GEMINI MODEL
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite-preview"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def extract_text(content):
    """
    Convert Gemini response content into plain text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# 4. JOB SEARCH TOOL
# ============================================================

@tool
def job_search(role: str) -> str:
    """
    Returns important skills required for a target job role.
    """

    role_lower = role.lower()

    if "python" in role_lower:

        return """
Python, FastAPI/Django, SQL, REST APIs,
Git/GitHub, Docker, Cloud basics
"""

    elif "java" in role_lower:

        return """
Java, OOP, Spring Boot, SQL,
REST APIs, Git/GitHub, DSA
"""

    elif "data analyst" in role_lower:

        return """
Python, SQL, Excel, Pandas,
NumPy, Power BI/Tableau, Statistics
"""

    elif (
        "machine learning" in role_lower
        or "ml" in role_lower
    ):

        return """
Python, SQL, Pandas, NumPy,
Scikit-learn, Machine Learning,
Statistics, Git
"""

    elif "frontend" in role_lower:

        return """
HTML, CSS, JavaScript, React,
Git/GitHub, REST APIs
"""

    elif "full stack" in role_lower:

        return """
HTML, CSS, JavaScript, React,
Node.js, Express.js, SQL/MongoDB,
Git/GitHub
"""

    else:

        return """
Programming fundamentals, DSA,
SQL, Git/GitHub, APIs,
problem solving
"""


# ============================================================
# 5. SKILL GAP TOOL
# ============================================================

@tool
def skill_gap_analysis(
    resume_skills: str,
    market_skills: str
) -> str:
    """
    Finds the most important missing skills.
    """

    prompt = f"""
Student skills:
{resume_skills}

Market skills:
{market_skills}

Find only the 2 or 3 most important missing skills.

Return only a short comma-separated list.
"""

    response = llm.invoke(prompt)

    return extract_text(
        response.content
    ).strip()


# ============================================================
# 6. PROJECT RECOMMENDATION TOOL
# ============================================================

@tool
def recommend_projects(
    skill_gaps: str
) -> str:
    """
    Recommends exactly two projects.
    """

    prompt = f"""
Skill gaps:
{skill_gaps}

Suggest exactly TWO suitable projects.

Format:

1. Project Name - one short description
2. Project Name - one short description

Keep the answer short.
"""

    response = llm.invoke(prompt)

    return extract_text(
        response.content
    ).strip()


# ============================================================
# 7. GITHUB PROFILE TOOL
# ============================================================

@tool
def github_profile_check(
    github_username: str
) -> str:
    """
    Checks a public GitHub profile.
    """

    headers = {}

    github_token = os.getenv(
        "GITHUB_TOKEN"
    )

    if github_token:

        headers["Authorization"] = (
            f"Bearer {github_token}"
        )

    # --------------------------------------------------------
    # User information
    # --------------------------------------------------------

    user_url = (
        f"https://api.github.com/users/"
        f"{github_username}"
    )

    try:

        user_response = requests.get(
            user_url,
            headers=headers,
            timeout=10
        )

    except Exception:

        return "GitHub profile could not be checked."

    if user_response.status_code != 200:

        return (
            f"GitHub username '{github_username}' "
            f"could not be found."
        )

    user = user_response.json()

    # --------------------------------------------------------
    # Repository information
    # --------------------------------------------------------

    repos_url = (
        f"https://api.github.com/users/"
        f"{github_username}/repos"
        f"?sort=pushed&per_page=10"
    )

    try:

        repos_response = requests.get(
            repos_url,
            headers=headers,
            timeout=10
        )

    except Exception:

        repos_response = None

    if (
        repos_response
        and repos_response.status_code == 200
    ):

        repos = repos_response.json()

    else:

        repos = []

    # --------------------------------------------------------
    # Analyze languages
    # --------------------------------------------------------

    languages = {}

    active_repos = 0

    for repo in repos:

        language = repo.get("language")

        if language:

            languages[language] = (
                languages.get(language, 0) + 1
            )

        pushed_at = repo.get(
            "pushed_at",
            ""
        )

        if pushed_at >= "2025-01-01":

            active_repos += 1

    top_languages = sorted(
        languages,
        key=languages.get,
        reverse=True
    )[:3]

    language_text = (
        ", ".join(top_languages)
        if top_languages
        else "None"
    )

    return (
        f"Repositories: "
        f"{user.get('public_repos', 0)}\n"
        f"Followers: "
        f"{user.get('followers', 0)}\n"
        f"Top Languages: "
        f"{language_text}\n"
        f"Recently Active Repositories: "
        f"{active_repos}"
    )


# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are PlacementPrep, an AI placement assistant.

The user provides:

- Resume skills
- Target role
- GitHub username

Perform these tasks:

1. Check important market skills for the target role.
2. Identify the most important skill gaps.
3. Recommend exactly 2 projects.
4. Check the GitHub profile.
5. Give a short placement report.

Use EXACTLY these sections:

### Market Skills
- Give 3-5 skills.

### Skill Gaps
- Give 2-3 missing skills.

### Projects
- Give exactly 2 projects.

### GitHub
- Give 1-2 short points.

### Next Steps
- Give exactly 3 short points.

IMPORTANT:

Keep the entire response under 15 lines.

Do not provide long explanations.
Do not provide tables.
Do not provide a 30-day plan.
Do not provide tool traces.
Do not return JSON.
Use simple bullet points.
"""


# ============================================================
# 9. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        job_search,
        skill_gap_analysis,
        recommend_projects,
        github_profile_check
    ],
    system_prompt=SYSTEM_PROMPT
)


# ============================================================
# 10. RUN AGENT
# ============================================================

def run_agent(input_data: dict) -> str:

    result = agent.invoke(
        input_data
    )

    messages = result.get(
        "messages",
        []
    )

    if not messages:

        return "No response generated."

    final_message = messages[-1].content

    return extract_text(
        final_message
    ).strip()


# ============================================================
# 11. SIMPLE INPUT
# ============================================================

def simple_input(
    text: str
) -> str:

    return run_agent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ]
        }
    )


clean_agent = RunnableLambda(
    simple_input
)


# ============================================================
# 12. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description="AI Placement Readiness Agent"
)


# ============================================================
# 13. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    clean_agent,
    path="/agent"
)


# ============================================================
# 14. WEB PAGE
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<title>Placement-Ready AI Agent</title>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    padding: 0;

    font-family: Arial, sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc
        );
}

.container {

    max-width: 750px;

    margin: 50px auto;

    background: white;

    padding: 35px;

    border-radius: 16px;

    box-shadow:
        0 8px 30px
        rgba(0, 0, 0, 0.10);
}

h1 {

    text-align: center;

    color: #1e293b;

    margin-bottom: 10px;
}

.subtitle {

    text-align: center;

    color: #64748b;

    margin-bottom: 30px;
}

label {

    display: block;

    margin-top: 20px;

    margin-bottom: 8px;

    font-weight: bold;

    color: #334155;
}

input[type="text"],
input[type="file"] {

    width: 100%;

    padding: 12px;

    border:
        1px solid #cbd5e1;

    border-radius: 8px;

    font-size: 15px;
}

input[type="file"] {

    background: #f8fafc;
}

button {

    width: 100%;

    margin-top: 25px;

    padding: 14px;

    border: none;

    border-radius: 8px;

    background: #2563eb;

    color: white;

    font-size: 16px;

    font-weight: bold;

    cursor: pointer;
}

button:hover {

    background: #1d4ed8;
}

button:disabled {

    background: #94a3b8;

    cursor: not-allowed;
}

#loading {

    display: none;

    text-align: center;

    margin-top: 20px;

    color: #2563eb;

    font-weight: bold;
}

#error {

    display: none;

    margin-top: 20px;

    padding: 15px;

    background: #fef2f2;

    color: #dc2626;

    border-radius: 8px;
}

#result {

    display: none;

    margin-top: 30px;

    padding: 20px;

    background: #f8fafc;

    border-radius: 10px;

    border:
        1px solid #e2e8f0;
}

#resumeSkills {

    padding: 15px;

    background: #eef2ff;

    border-radius: 8px;

    line-height: 1.6;

    margin-bottom: 25px;
}

#report {

    white-space: pre-wrap;

    line-height: 1.6;

    color: #334155;
}

h2 {

    color: #1e293b;
}

</style>

</head>


<body>


<div class="container">

<h1>
🎯 Placement-Ready AI Agent
</h1>


<p class="subtitle">

Upload your resume and get a
short placement-readiness analysis.

</p>


<!-- Resume -->

<label>
📄 Resume PDF
</label>

<input
    type="file"
    id="resume"
    accept=".pdf"
>


<!-- Target Role -->

<label>
💼 Target Role
</label>

<input
    type="text"
    id="role"
    placeholder="Example: Python Developer"
>


<!-- GitHub Username -->

<label>
🐙 GitHub Username
</label>

<input
    type="text"
    id="github"
    placeholder="Example: Harika123"
>


<!-- Analyze -->

<button
    id="analyzeButton"
    onclick="analyzeResume()"
>

Analyze Resume

</button>


<!-- Loading -->

<div id="loading">

⏳ Analyzing your resume...
Please wait.

</div>


<!-- Error -->

<div id="error"></div>


<!-- Result -->

<div id="result">

<h2>
📋 Resume Skills
</h2>

<div id="resumeSkills"></div>


<h2>
📊 Placement Report
</h2>

<div id="report"></div>

</div>


</div>


<script>


async function analyzeResume() {

    const resume =
        document.getElementById(
            "resume"
        ).files[0];

    const role =
        document.getElementById(
            "role"
        ).value.trim();

    const github =
        document.getElementById(
            "github"
        ).value.trim();


    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    if (!resume) {

        showError(
            "Please choose your resume PDF."
        );

        return;
    }


    if (
        !resume.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        showError(
            "Please upload a PDF file."
        );

        return;
    }


    if (!role) {

        showError(
            "Please enter your target role."
        );

        return;
    }


    if (!github) {

        showError(
            "Please enter your GitHub username."
        );

        return;
    }


    // --------------------------------------------------------
    // Form Data
    // --------------------------------------------------------

    const formData =
        new FormData();


    formData.append(
        "resume",
        resume
    );

    formData.append(
        "role",
        role
    );

    formData.append(
        "github_username",
        github
    );


    // --------------------------------------------------------
    // UI
    // --------------------------------------------------------

    const button =
        document.getElementById(
            "analyzeButton"
        );

    button.disabled = true;


    document.getElementById(
        "loading"
    ).style.display = "block";


    document.getElementById(
        "error"
    ).style.display = "none";


    document.getElementById(
        "result"
    ).style.display = "none";


    // --------------------------------------------------------
    // API Request
    // --------------------------------------------------------

    try {

        const response =
            await fetch(
                "/analyze",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            showError(
                data.detail ||
                data.error ||
                "Something went wrong."
            );

            return;
        }


        // ----------------------------------------------------
        // Resume Skills
        // ----------------------------------------------------

        document.getElementById(
            "resumeSkills"
        ).innerText =
            data.resume_skills;


        // ----------------------------------------------------
        // Report
        // ----------------------------------------------------

        document.getElementById(
            "report"
        ).innerText =
            data.report;


        document.getElementById(
            "result"
        ).style.display =
            "block";


    } catch (error) {

        showError(
            "Server connection error: "
            + error.message
        );

    } finally {

        button.disabled = false;

        document.getElementById(
            "loading"
        ).style.display =
            "none";
    }
}


// ============================================================
// SHOW ERROR
// ============================================================

function showError(message) {

    const error =
        document.getElementById(
            "error"
        );

    error.innerText =
        message;

    error.style.display =
        "block";
}


</script>


</body>

</html>
"""


# ============================================================
# 15. HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE


# ============================================================
# 16. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 17. RESUME ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_username: str = Form(...)
):

    # --------------------------------------------------------
    # Validate resume
    # --------------------------------------------------------

    if not resume.filename:

        return {
            "error":
                "Resume file is required."
        }


    if not resume.filename.lower().endswith(
        ".pdf"
    ):

        return {
            "error":
                "Only PDF files are supported."
        }


    # --------------------------------------------------------
    # Save uploaded PDF
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temporary_file:

        file_data = await resume.read()

        temporary_file.write(
            file_data
        )

        temporary_path = (
            temporary_file.name
        )


    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    try:

        reader = PdfReader(
            temporary_path
        )

        resume_text = ""

        for page in reader.pages:

            resume_text += (
                page.extract_text()
                or ""
            )

    finally:

        if os.path.exists(
            temporary_path
        ):

            os.unlink(
                temporary_path
            )


    # --------------------------------------------------------
    # Extract technical skills
    # --------------------------------------------------------

    skills_prompt = f"""
Extract ONLY technical skills
from this resume.

Return ONLY a comma-separated list.

Do not include:

- Name
- Email
- Phone
- Address
- Education
- Soft skills

Resume:

{resume_text[:8000]}
"""


    skills_response = llm.invoke(
        skills_prompt
    )


    resume_skills = extract_text(
        skills_response.content
    ).strip()


    # --------------------------------------------------------
    # Agent input
    # --------------------------------------------------------

    user_input = f"""
Target Role:
{role}

Resume Skills:
{resume_skills}

GitHub Username:
{github_username}

Give a short placement readiness report.
"""


    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    final_report = clean_agent.invoke(
        user_input
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {
        "resume_skills":
            resume_skills,

        "report":
            final_report
    }


# ============================================================
# 18. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )
