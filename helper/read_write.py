import os
from networkx.drawing.nx_agraph import write_dot
import networkx as nx
import os
import operator
import scipy.stats as sci
import timeit
import numbers
import numpy
import unicodecsv as csv
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_files(path='', fmt='graphml'):
    """
    :param path: optional parameter , path to the directory where files are to me read from.
    :param fmt: optional parameter, extension of the files that are to be selected. default graphml
    :return: the list of file names with full path
    """
    if path is '':
        path = os.getcwd() + '/data'
    return [path + '/' + f for f in os.listdir(path) if f.endswith('.' + fmt)]


def create_result_file(extension, result_folder='result', input_file='result.someext'):

    """
    :param extension: extension of the filename needed example .pdf , .graphml, .xml
    :param path: (optional) path where result folder is to be created, if the path is blank , current working directory will be used to create the result folders
    :param rel_result_path: folder structure of the result data eg: result/  will create a folder "result" in path
    :param filename
    :return the absolute file name eg "path/rel_result_path/filename.extension"

    """
    dirname, filename = os.path.split(os.path.abspath(input_file))
    new_name = filename.rpartition('.')[0] + '.' + extension

    new_dir = dirname + '/' + result_folder + '/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    out_file = new_dir + new_name
    return out_file


def change_extension(ext, filename):
    return filename.rpartition('.')[0] + ext


def change_name_and_ext(ext, filename, newname):
    return filename.rpartition('.')[0] + '_' + newname + '.' + ext

def write_dot_output(graphml_file, outfile=''):
    """
    This function takes the graphml file and writes the dot file for visualization

    :param graphml_file: Input file in graphml format
    :param outfile: Output file name
    :return: path of output file
    """
    graph = nx.read_graphml(graphml_file) # read the graphml file to a graph

    # use the infile name and path if no outfile is specified
    if outfile == '':
       outfile = change_extension('.dot', graphml_file)
    write_dot(graph, outfile)
    print('dot file written to {0}'.format(outfile))

def draw_and_save(G, filename='draw.gml.gz'):
    nx.write_graphml(G, filename)

def write_graph(graph, filename, name, folder='/results/'):
    """

    :param graph:
    :param filename:
    :param name:
    :param folder:
    :return:
    """
     # put the subgraph output in a folder results
    dirname, filename = os.path.split(os.path.abspath(filename))
    new_name = filename.rpartition('.')[0] + '_' + name + '.graphml'
    newdir = dirname.rpartition('/')[0] + folder + name +'/'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    out_file = newdir + new_name
    nx.write_graphml(graph, out_file)


def visualize_results(sorted_dict, file, name='_figure.pdf'):
    """

    :param graph:
    :param file:
    :return:
    """

    # put the pdf output in a folder results
    dirname, filename = os.path.split(os.path.abspath(file))
    new_name = filename.rpartition('.')[0] + name
    newdir = dirname.rpartition('/')[0] + '/results/pr/graphs/'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    out_figure = newdir + new_name

    # x = [i for i in range(1, len(sorted_dict) + 1)]
    pr = [value["rank"] for k, value in sorted_dict]

    # attributes = {}
    attribute_list = {}
    first = True
    for k, value in sorted_dict:
        if first:
            attribute_list['user'] = []
        attribute_list['user'].append(k)
        for key, val in value.iteritems():
            if isinstance(val, numbers.Number):
                if first:
                    attribute_list[key] = []
                attribute_list[key].append(val)
        first = False

    pdf = PdfPages(out_figure)
    for attribute, value in attribute_list.iteritems():
        plot_graph(pdf, attribute, value, pr)



    pdf.close()


def plot_graph(pdf, attr_name, attr_list, page_rank, label2="page-rank"):
    """

    :param pdf:
    :param attr_name:
    :param attr_list:
    :param page_rank:
    :return:
    """
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    attr_c = host.twinx()

    attr_label = ' '.join(attr_name.split('_'))

    x = [i for i in range(1, len(page_rank) + 1)]
    p1, = host.plot(x[:200], page_rank[:200], "b-", label=label2)
    p2, = attr_c.plot(x[:200], attr_list[:200], "r-", label=attr_label)

    host.set_xlabel("Users (sorted by page rank)")
    host.set_ylabel(label2)
    attr_c.set_ylabel(attr_label)

    host.yaxis.label.set_color(p1.get_color())
    attr_c.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    attr_c.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2]
    plt.title(" Plot of {0} and {1}\n"
              " Pearson co-relation value is {2:04.2f}\n"
              " Spearmans Correlation value is {3:04.2f}".format(label2, attr_label,
                                                                 numpy.corrcoef(page_rank, attr_list)[0, 1],
                                                                 sci.spearmanr(page_rank,attr_list).correlation))

    print(sci.spearmanr(page_rank,attr_list))

    host.legend(lines, [l.get_label() for l in lines])
    pdf.savefig()
    plt.close()




def write_user_data(sorted_dict, file, name='_user_data.csv'):
    # put the pdf output in a folder results
    dirname, filename = os.path.split(os.path.abspath(file))
    new_name = filename.rpartition('.')[0] + name
    newdir = dirname.rpartition('/')[0] + '/results/pr/user_data/'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    out_file = newdir + new_name

    with open(out_file, 'wb') as f:
        writer = csv.writer(f, encoding='utf-8')
        i = 0
        for key, value in sorted_dict:
            print (key)
            print (value)
            if i == 0:
                print(['User Id'] + value.keys())
                writer.writerow(['User Id'] + value.keys())
            i += 1
            print([key] + value.values())
            writer.writerow([key] + value.values())
            if i == 20:
                break

