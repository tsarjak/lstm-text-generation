def remove(str_, chars):
	table = {ord(char): None for char in chars}
	return str_.translate(table)

def main():
	completeText = open("Unprocessed.txt").read().lower()

	# To remove unneccesary characters from the string - define a list
	unWanted = ['\r','_','“','”']
	completeText = remove(completeText, unWanted)

	# To replace some characters to reduce the vocabulary
	completeText = completeText.replace(';', ',')
	completeText = completeText.replace(':', '-')
	completeText = completeText.replace('"', "'")
	completeText = completeText.replace('\n', ' ')

	with open("Processed.txt", "w") as txtFile:
		txtFile.write("%s" % completeText)


if __name__ == '__main__':
	main()
