from PCA import train, interpolate_to_grid, closest_int
from tqdm import tqdm
from parser import Parser
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import re
import os
import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_on_mixtures(do_training: bool = True) -> None:
    file = os.path.join(
                os.path.dirname(__file__),
                "../../data/scrambled/pure/Primpke_FTIR_Microplastics_converted_scrambled_pure.csv"
            )
    if do_training:
        succeeded = train(
            data_base_path = file,
            output_path = os.path.join(
                os.path.dirname(__file__),
                "../../data/test_matrix.mtx"
            ),
            min_wavenumber = 400,
            max_wavenumber = 4000,
            step_size=2.
        )
        if not succeeded:
            logger.error("Was unable to train")
            return 

    parser = Parser()
    training_file, U, S, Vt, filtered_grid = parser.parse(os.path.join(os.path.dirname(__file__), "../../data/test_matrix.mtx"))
    print(U.shape, S.shape, Vt.shape, filtered_grid.shape)
    
    PCA_matrix = (U*S.squeeze())@Vt
    np.nan_to_num(PCA_matrix, nan=0.)
    scores = U*S.squeeze()

    print("Scores", scores.shape)
      
    mixed_file = file.replace("pure", "mixed")
    mixed_data = pd.read_csv(mixed_file, dtype=str)
    errors = 0
    loss = 0
    unprocessable = 0

    entry_names = pd.read_csv(training_file, header=None, nrows = 1).iloc[0].to_numpy()
    entry_names = entry_names[pd.notna(entry_names)]

    with tqdm(total=mixed_data.shape[1]//2 - 1, desc="Processing mixtures") as pbar:
        for i in range(0, mixed_data.shape[1]-2, 2):
            mixed_entry_name = mixed_data.axes[-1][i+1]
            mixed_entry = mixed_data.iloc[2:, i:i + 2]
            mixed_waveNumber = np.array(mixed_entry.iloc[:, 0].astype(float))
            mixed_absorbance = np.array(mixed_entry.iloc[:, 1].astype(float))
    #         print("SIZES = ", mixed_waveNumber.size, mixed_absorbance.size, filtered_grid.size)
    #         print(
    #             mixed_entry_name,
    #             "total:", len(mixed_waveNumber),
    #             "NaN wn:", np.isnan(mixed_waveNumber).sum(),
    #             "NaN absorbance:", np.isnan(mixed_absorbance).sum(),
    #             "valid:", np.sum(
    #                 ~np.isnan(mixed_waveNumber) &
    #                 ~np.isnan(mixed_absorbance)
    #             )
    #         )
            
            mixture = interpolate_to_grid(mixed_waveNumber, mixed_absorbance, filtered_grid).astype(float)
            np.nan_to_num(mixture, nan=0.)
            mixture_scores = (Vt@mixture).T
            
            try: 
                coefficients, residual_norm = nnls(
                    PCA_matrix.T,
                    mixture
                )
            except:
                unprocessable += 1
                continue

            coefficients /= coefficients.sum()
            # TODO: use a meaningful threshhold
            threshhold = 0.05 # 5%
            coefficients = np.array([coefficient if coefficient > threshhold else 0 for coefficient in coefficients])
            coefficients /= coefficients.sum()
                
            sorted_indices = np.argsort(coefficients)[::-1]

            predictions = entry_names[sorted_indices[:2]]
            prediction_1 = re.match(r'^(.*?)_(\d*?)?', predictions[0]).groups()[0]
            prediction_2 = re.match(r'^(.*?)_(\d*?)?', predictions[1]).groups()[0]

            selected_indices = []
            seen_names = set()
            

            for idx in sorted_indices:
                base_name = re.sub(r'_\d+$', '', entry_names[idx])

                if base_name not in seen_names:
                    seen_names.add(base_name)
                    selected_indices.append(idx)

                if len(selected_indices) == 3:
                    break

            first, second = mixed_entry_name.split(' + ')
            
            firstName, firstPercent = re.match(r'^(.*?)_(\d*?)%$', first).groups()
            secName, secPercent = re.match(r'^(.*?)_(\d*?)%$', second).groups()
            
            print(prediction_1, firstName, prediction_2, secName) 

            idxFirst = [idx for idx in sorted_indices if re.match(firstName, entry_names[idx])]
            idxSec = [idx for idx in sorted_indices if re.match(secName, entry_names[idx])]

            if idxFirst and idxSec:
                foundFirstPercent = coefficients[idxFirst[0]]*100
                foundSecPercent = coefficients[idxSec[0]]*100

                loss += ((int(firstPercent) - foundFirstPercent)**2 + (int(secPercent) - foundSecPercent)**2)**0.5
                errors += int(prediction_1 not in (firstName, secName)) + int(prediction_2 not in (firstName, secName))
            else: 
                unprocessable += 1

            pbar.set_postfix({"loss" : f'{loss:.2f}', "unprocessable":f'{unprocessable}', "Errors":f'{errors}'})
            pbar.update(1)
        print(f'Total loss of file: {loss}\n Mean loss of file: {loss/(mixed_data.shape[1]//2 - 1)}') 
        print(f"Errors made (didn't predict the top 2 correctly): {errors} ~ {errors/2*(mixed_data.shape[1]//2 - 1)}%")


test_on_mixtures(False)
