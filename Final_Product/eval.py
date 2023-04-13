def read_correct(correct_txt_path):
    file = open(correct_txt_path, "r")
    return [line.strip() for line in file.readlines()]


def read_actual(actual_txt_path):
    file = open(actual_txt_path, "r")
    return [line.strip() for line in file.readlines()]


def eval(correct_ids, actual_ids):
    correct = 0
    for id in correct_ids:
        if id in actual_ids:
            correct += 1
        else:
            print(id)

    return correct / len(correct_ids) * 100


if __name__ == "__main__":
    correct_path = "./Final_Product/Correct_Plates.txt"
    correct_ids = read_correct(correct_path)
    actual_path = "./Final_Product/cropped_plates/Plates_Only.txt"
    actual_ids = read_actual(actual_path)

    score = eval(correct_ids, actual_ids)

    print(score)
