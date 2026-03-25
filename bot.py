import sys
from pipeline import MediRAG


def main():
    print("MediRAG — Health Assistant")
    print("type your question or /exit to quit\n")

    try:
        bot = MediRAG()
    except FileNotFoundError as e:
        print(f"error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)

    show_sources = True

    while True:
        try:
            user_input = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("bye")
            break
        elif user_input == "/sources":
            show_sources = not show_sources
            print(f"sources: {'on' if show_sources else 'off'}\n")
            continue

        print("searching...\n")

        try:
            result = bot.ask(user_input)
            print("MediRAG:")
            print(result.answer)

            if result.diseases:
                print(f"\ndiseases found: {', '.join(result.diseases)}")

            if show_sources:
                print("sources:")
                for s in result.sources():
                    print(f"  - {s}")

            print()

        except Exception as e:
            print(f"error: {e}\n")


if __name__ == "__main__":
    main()