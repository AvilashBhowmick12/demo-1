from src.shortlist import ResumeShortlister
from src.chat import ChatShortlister

HELP = """Describe the role and skills you want, then keep refining. Examples:
  I need a DevOps engineer with AWS and Terraform, bonus for Kubernetes
  only those with Docker
  add python as a must-have
  show 20 results
  why is #2 ranked above #3?
  compare the top two
Commands: /reset (start over), /help, exit"""

if __name__ == "__main__":
    shortlister = ResumeShortlister()
    shortlister.ensure("data")   # loads the saved index, or builds it on the first run
    chat = ChatShortlister(shortlister)

    print("\nResume shortlister ready.\n" + HELP + "\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("exit", "quit"):
            break
        if message == "/help":
            print(HELP)
            continue
        if message == "/reset":
            chat.reset()
            print("Started over.")
            continue
        if message:
            print("\n" + chat.handle(message) + "\n")