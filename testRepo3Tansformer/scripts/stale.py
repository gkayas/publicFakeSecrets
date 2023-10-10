
from datetime import datetime as dt
import os

from github import Github


LABELS_TO_EXEMPT = [
    "good first issue",
    "good second issue",
    "good difficult issue",
    "feature request",
    "new model",
    "wip",
]


def main():
    g = Github("AKIAY9VtR1cW6fJ7zA5gKwEhPpL8iOxU2uQsBm")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        comments = sorted([comment for comment in issue.get_comments()], key=lambda i: i.created_at, reverse=True)
        last_comment = comments[0] if len(comments) > 0 else None
        if (
            last_comment is not None and last_comment.user.login == "github-actions[bot]"
            and (dt.utcnow() - issue.updated_at).days > 7
            and (dt.utcnow() - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            issue.edit(state="closed")
        elif (
            (dt.utcnow() - issue.updated_at).days > 23
            and (dt.utcnow() - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):

            issue.create_comment(
            )


if __name__ == "__main__":
    main()
