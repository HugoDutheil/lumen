from lumen.parser import Parser
from lumen.PCA import train, interpolate_to_grid, closest_int
import pandas as pd
import os

def benchmark_PCA():
    
    results_file = os.path.join(
            os.path.dirname(__file__),
            "../../data/scrambled/mixed/Primpke_FTIR_Microplastics_converted_scrambled_pure.csv"
        )
    expected_results = pd.read_csv(results_file)
    formated_expected_values = pd.DataFrame(columns=["first_name", "first_percentage", "second_name", "second_percentage"])

    names = expected_results.head(1)
    for name in names:
        first, second = name.split(" + ")
        name1, percent1 = re.match(r'^(.*?)_(\d*?)?', first).groups()
        name2, percent2 = re.match(r'^(.*?)_(\d*?)?', second).groups()
        formated_expected_values.loc[len(formated_expected_values)] = [name1, percent1, name2, percent2] 


    @becnhmark(expected_results=formated_expected_values)
    def PCA_pipeline(do_training: bool = True) -> None:
        
        output_predictions = pd.DataFrame(columns=["first_name", "first_percentage", "second_name", "second_percentage"])

        file = os.path.join(
            os.path.dirname(__file__),
            "../../data/scrambled/pure/Primpke_FTIR_Microplastics_converted_scrambled_pure.csv"
        )

        matrix_loc = os.path.join(
            os.path.dirname(__file__),
            "../../data/test_matrix.mtx"
        )

        if do_training:
            succeeded = train(
                data_base_path = file,
                output_path = matrix_loc,
                min_wavenumber = 400,
                max_wavenumber = 4000,
                step_size=2.
            )
            if not succeeded:
                logger.error("Was unable to train")
                return 

        parser = Parser()
        training_file, U, S, Vt, filtered_grid = parser.parse(matrix_loc)
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
                
                mixture = interpolate_to_grid(mixed_waveNumber, mixed_absorbance, filtered_grid).astype(float)
                np.nan_to_num(mixture, nan=0.)
                mixture_scores = (Vt@mixture).T
                
                try: 
                    coefficients, residual_norm = nnls(
                        PCA_matrix.T,
                        mixture
                    )
                except:
                    unprocessable.append(i)
                    prediction =  [np.nan]*4
                    output_predictions.loc[len(output_predictions)] = prediction
                    continue

                coefficients /= coefficients.sum()
                # TODO: use a meaningful threshhold
                threshhold = 0.05 # 5%
                coefficients = np.array([coefficient if coefficient > threshhold else 0 for coefficient in coefficients])
                coefficients /= coefficients.sum()
                
                # Sort to get the list of what it considers most likely
                sorted_indices = np.argsort(coefficients)[::-1]
                
    #             # Take top two
    #             predictions = entry_names[sorted_indices[:2]]
    #             prediction_1 = re.match(r'^(.*?)_(\d*?)?', predictions[0]).groups()[0]
    #             prediction_2 = re.match(r'^(.*?)_(\d*?)?', predictions[1]).groups()[0]

                merged_names: dict = dict() 

                # Doing this so it doesn't start predicting elementA_1 elementA_2, etc.. and only considers to have seen elementA without the scramble prefix. TODO: adapt this to general for text 
                for idx in sorted_indices:
                    base_name = re.sub(r'_\d+$', '', entry_names[idx])

                    if base_name not in merged_names:
                        merged_names[base_name] = coefficients[idx]
                    else:
                        merged_names[base_name] += coefficients[idx]
                
                df = pd.DataFrame(merged_names.items(), columns=["name", "value"])
                df["values"] /= df["values"].sum()

                            
                prediction =  [df.iloc[0].name, df.iloc[0].value, df.iloc[1].name, df.iloc[1].value]
                output_predictions.loc[len(output_predictions)] = prediction
            
            return output_predictions, unprocessable
    return


benchmark_PCA()
