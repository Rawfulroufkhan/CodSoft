def chatbot():
    print("Hello Sir, How Can I Get Your Order?")

    def menu():
        print("Bot: Here is the menu:")
        for item in menu_list:
            print(item)

    menu_list = ["Burger", "Pizza", "Fried Rice", "Chicken Fry", "Noodles"]

    while True:
        user_input = input("You: ").lower()
        if "yes" in user_input:
            print("Bot: Do You Want Menu?")
            user_input2 = input("You: ").lower()
            if "yes" in user_input2:
                menu()
                user_input3 = input("You: ").lower()
                if user_input3 in [item.lower() for item in menu_list]:
                    print(f"Bot: What Kind Of {user_input3.capitalize()} You Want?")
                    user_input4 = input("You: ").lower()
                    print("Bot: Do You Want To Confirm Your Order?")
                    user_input5 = input("You: ").lower()
                    if "yes" in user_input5:
                        print("Bot: Please Kindly Give Your Details!!")
                        user_input_name = input("Name: ")
                        user_input_phone = int(input("Phone: "))
                        user_input_Address = input("Address: ")
                        print(f"Bot: \nDear {user_input_name} Sir ,\nYour Order for {user_input4} {user_input3}, at {user_input_Address} is Confirmed!!\nYou Will Get The Delivery in 30 Minutes \nOur Deliveryman Will Contact With You on {user_input_phone} This Number\nThanks For Visiting Us")
                    elif "no" in user_input5:
                        print("Bot: Do You Want to Order More?")
                        user_input6=str(input("You: ")).lower()
                        if "yes" in user_input6:
                            menu()
                        elif "no" in user_input6:
                            print("Bot: Thanks For Visiting Us.\nHope To See You Again")
                        else:
                            print("Bot: Can you rephrase?")
                    else:
                        print("Bot: Can you rephrase?")
                    break
                else:
                    print("Bot: Can you rephrase?")
                break
            elif "no" in user_input2:
                print("Bot: What Do You Want To Order?")
                user_input_nomenu = input("You: ").lower()
                if user_input_nomenu in [item.lower() for item in menu_list]:
                    print("Bot: This Item is Already in Your Menu!\nSelect From The Menu Please")
                    menu()
                else:
                    print(f"Bot: {user_input_nomenu.capitalize()} is not in our menu.Please Select From Our Menu")
            else:
                chatbot
        elif "no" in user_input:
            print("Bot: Then How Can I Help You Sir?")
            user_input_other_help=str(input("You: ")).lower()
            if "help" in user_input_other_help:
                print("Bot: Please call at 999000111")
            elif "location" in user_input_other_help:
                print("Bot: We Have Branches at\nDhaka\nKolkata\nMumbai\nNew york")
            elif "complain" in user_input_other_help:
                print("Bot: I am so sorry for your trouble\nPlease file the complain in details")
                user_input_complain=str(input("You: "))
                print("Bot: Your complain has been recorded \nwe will reach to you soon")
            else:
                print("Bot: Can you rephrase?")
        elif user_input == "exit":
            print("Bot: See you later!")
            break
        else:
            print("Bot: Please rephrase")
        

chatbot()
