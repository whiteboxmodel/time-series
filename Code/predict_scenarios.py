import pandas as pd
import matplotlib.pyplot as plt
from make_sequences import *


# Load preprocessed scenario files
def load_scenarios(scenario_files, start_date):
    d_scenarios = {}
    for scenario in scenario_files.keys():
        d = pd.read_csv(scenario_files[scenario])
        d = d[d['date'] >= start_date].reset_index(drop = True)
        d_scenarios[scenario] = d
    return d_scenarios


def predict_scenarios(scenario_data, x_columns, y_columns, model):
    d_forecast = pd.DataFrame()
    for scenario in scenario_data.keys():
        d = scenario_data[scenario]
        d_forecast[[scenario + '_FRB']] = d[y_columns] # Keep the original FRB series
        x_scenario = create_fixed_length_sequences(d[x_columns], sequence_length = d.shape[0])[0]
        y_scenario = model(x_scenario)
        y_scenario = y_scenario[0, :, 0].detach().numpy()
        d_forecast[scenario] = y_scenario
    d_forecast.insert(0, 'date', d['date'])
    return d_forecast


def plot_scenario_forecasts(d_forecast, y_label):
    # Base scenario
    d_forecast.plot(x = 'date', y = ['Base_FRB', 'Base'], grid = True, rot = 45,
                    xlabel = '', ylabel = y_label, title = 'Base scenario')
    plt.tight_layout()
    plt.show()
    # Severely adverse scenario
    d_forecast.plot(x = 'date', y = ['SA_FRB', 'SA'], grid = True, rot = 45,
                    xlabel = '', ylabel = y_label, title = 'Severely adverse scenario')
    plt.tight_layout()
    plt.show()
