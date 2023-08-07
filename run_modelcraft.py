import os
import subprocess
from multiprocessing import Pool
import tqdm


def get_test_list(test_list_path: str):
    test_list = []

    with open(test_list_path, "r") as output_file:
        for line in output_file:
            test_list.append(line.strip("\n"))

    return test_list


def worker(pdb):
    base_dir = "rna_test/tests/base"
    work_dir = os.path.join(base_dir, pdb)
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

    contents_path = f"../../../data/contents/{pdb}.json"
    mtz_path = f"../../../data/mtz_simulated/simulated_{pdb}.mtz"

    if not os.path.isfile(mtz_path):
        return

    subprocess.run(["modelcraft", "xray", "--contents", contents_path, "--data", mtz_path, "--phases", "PHIC,FOM"],
                   cwd=work_dir)


def new_worker(pdb):
    base_dir = "rna_test/tests/2.4.1_base"
    work_dir = os.path.join(base_dir, pdb)

    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

    log_path = os.path.join(work_dir, f"{pdb}.log")

    contents_path = f"../../../data/contents/{pdb}.json"
    mtz_path = f"/home/jordan/dev/modelcraft/rna_test/data/mtz_simulated/simulate_{pdb}.mtz"

    if not os.path.isfile(mtz_path):
        print(mtz_path, "not found")
        return

    subprocess.run(["modelcraft", "-v"])

    with open(log_path, "w") as log_file:
        subprocess.run(
            [
                # "python", "-m", 
            "modelcraft", "xray", "--contents", contents_path, "--data", mtz_path, "--phases", "simulate.ABCD.A,simulate.ABCD.B,simulate.ABCD.C,simulate.ABCD.D"
             , "--observations", "simulate.F_sigF.F,simulate.F_sigF.sigF", "--keep-files", "--cycles", "1"],
            # , "--disable-parrot"],
            cwd=work_dir, stdout=log_file)


def main():
    test_list = get_test_list("rna_test/data/test_list.txt")

    for x in test_list:
        new_worker(x)

    # new_worker(test_list[7])
    
    # with Pool() as pool:
    #     x = list(tqdm.tqdm(pool.imap_unordered(new_worker, test_list), total=len(test_list)))


if __name__ == "__main__":
    main()
