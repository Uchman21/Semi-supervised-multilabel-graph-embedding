import re
import os



def extract_best(dataset, files, output_file, is_supervised, method):
	best = -1
	best_config = []
	prog = re.compile(r"(?<=\[).*(?=\])")
	for file in files:
		result = open("outputs/"+file, 'rU').read().split("---------------------best general --------------------\n")[-1].split("\n")
		try:
			print(result[0])
			if result[0] and result[0]!= "":
				acc = float(prog.search(result[0]).group(0))
				if best < acc:
					best = acc
					best_config = [prog.search(result[1]).group(0)]
				elif best == acc:
					best_config.append(prog.search(result[1]).group(0))
		except:
			pass

	if best > 0:
		output_file.write("{}\t{}\t{}\t{}\t{}\n".format(dataset, method, is_supervised, best, best_config))


def main():
	doc_list = os.listdir("outputs/")
	datasets = ["cora", "citeseer", "pubmed"]#, "dblp"]
	_types = ["cluster", "vae"]
	is_supervised = ["True", "False"]
	mc_result = open("multi_class_result.txt", "w")
	ml_result = open("multi_label_result.txt", "w")


	for dataset in datasets:
		for _type in _types:
			for l_method in is_supervised:
				prog = re.compile("{}_{}.*_{}".format(dataset,_type,l_method))
				dataset_files = [ file for file in doc_list if prog.search(file)]
				if dataset != "dblp":
					extract_best(dataset, dataset_files, mc_result, l_method, _type)
				else:
					extract_best(dataset, dataset_files, ml_result, l_method,_type)


if __name__ == '__main__':
	main()

