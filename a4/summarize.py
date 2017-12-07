"""
Writes all the outputs to SUMMARY.TXT file - Takes the cluster and classify outputs.
"""


def main():
    """
    main method
    """

    f1 = open('summary.txt', 'w')

    f = open('cluster_results.txt', 'r')
    f1.write(f.read())
    f.close()

    f = open('classify_results.txt', 'r')
    f1.write(f.read())
    f.close()


if __name__ == '__main__':
    main()
