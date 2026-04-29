
import json
import sys


def get_response(user_input):
    return "This is a response from the model based on your input: " + user_input


def main():
    user_input = sys.argv[1] if len(sys.argv) > 1 else ""
    response = get_response(user_input)
    print(json.dumps({"response": response}))


if __name__ == "__main__":
    main()