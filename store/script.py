def run_my_script():
    print("Script is running!")
    with open("output.txt", "w") as f:
        f.write("Script executed successfully!")
