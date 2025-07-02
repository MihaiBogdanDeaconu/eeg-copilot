import os

def main():
    """
    Provides instructions for downloading the TUH EEG Corpus.
    """
    print("=====================================================================")
    print("AI Co-Pilot: TUH EEG Corpus Download Instructions")
    print("=====================================================================")
    print("The TUH EEG Corpus is a massive, invaluable resource for EEG research, but it requires manual registration and download.")
    print("\n--- Step-by-Step Instructions ---")
    print("1. Go to the NEDC website: https://www.nedcdata.org/")
    print("2. Click on 'Data' and navigate to the 'TUH EEG Corpus'.")
    print("3. You will need to register for an account. This is a manual process that requires approval from the administrators. Fill out the forms accurately.")
    print("4. Once your account is approved, log in and go to the download section for the TUH EEG Corpus (v2.0.0 or later is recommended).")
    print("5. Download the dataset. It is very large (~1.5 TB), so ensure you have sufficient disk space and a stable internet connection. You can start with a smaller subset if needed.")
    print("6. After downloading, unzip the files into a directory.")
    print("\n--- Final Step ---")
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'TUH_EEG')
    print(f"7. Place the entire dataset into the following directory: {data_dir}")
    print("   (You may need to create the 'data/TUH_EEG' directories).")
    print("8. Finally, update the `paths.tuh_eeg_dir` key in `config/config.yaml` to point to this directory.")
    print("\nThis script does not perform the download automatically due to the access restrictions.")
    print("=====================================================================")

if __name__ == "__main__":
    main()