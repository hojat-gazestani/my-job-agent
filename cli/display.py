from models.models import JobReport


def render_markdown(report: JobReport) -> str:
    lines = [
        "# Job Matching Evaluation Report",
        "",
        "## Top matches",
        "",
        "| Job Title | Company | Location | Fit score | Visa Status | Link |",
        "|---|---|---|---|---|---|",
    ]

    for job in sorted(report.jobs, key=lambda x: x.fit_score, reverse=True):
        lines.append(
            f"| {job.title} | {job.company} | {job.location} | "
            f"{job.fit_score}% | {job.visa_status} |"
            f"[View Job]({job.url}) |"
        )

    lines.extend(["", "## Detailed Breakdown", ""])

    for i, job in enumerate(
        sorted(report.jobs, key=lambda x: x.fit_score, reverse=True),
        1,
    ):
        lines.extend(
            [
                f"### {i}. {job.title}",
                f"- **Company:** {job.company}",
                f"- **Location:** {job.location}",
                f"- **Fit Score:** {job.fit_score}",
                f"- **Visa Status:** {job.visa_status}",
                f"- **Reason:** {job.reason}",
                "",
            ]
        )
    return "\n".join(lines)
