import os
import parseidf
import re
import numpy as np
from eppy.modeleditor import IDF
import ntpath
from shutil import copy
import esoreader
import pandas as pd
import time
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import plotly.io as pio
import webbrowser
import operator


class idf_simulator:

    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
        '#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#bae1ff','red',
        'green', 'blue', 'orange', 'black', 'aquamarine', 'azure',
        'beige', 'bisque', 'blanchedalmond', 'blue',
        'blueviolet', 'brown', 'burlywood', 'cadetblue',
        'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey',
        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
        'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
        'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
        'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
        'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
        'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
        'lightgoldenrodyellow', 'lightgray', 'lightgrey',
        'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
        'lightskyblue', 'lightslategray', 'lightslategrey',
        'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
        'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple',
        'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
        'mediumturquoise', 'mediumvioletred', 'midnightblue',
        'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
        'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
        'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
        'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
        'plum', 'powderblue', 'purple', 'red', 'rosybrown',
        'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon',
        'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
        'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
        'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
        'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
        'yellow', 'yellowgreen']

    material_cost = {
        'm182q23': 823.25,
        'm182q25': 10557,
        'm182q27': 10557,
        'm182q29': 422.21,
        'm182q29concentrated': 422.21,
    }

    materials_props_lookup = {
        "m182q29": ['Material', 'BioPCM M182/Q29', 'VeryRough', '0.0742', '0.2', '235', '1970', '0.9', '0.1',
                    '0.5'],
        "m182q29concentrated": ['Material', 'BioPCM M182/Q29 concentrated', 'VeryRough', '0.0742', '0.16',
                                '850',
                                '2500', '0.9', '0.1', '0.5'],
        "m182q23": ['Material', 'BioPCM M182/Q23', 'VeryRough', '0.0742', '0.2', '235', '1970', '0.9', '0.1',
                    '0.5'],
        "m182q25": ['Material', 'BioPCM M182/Q25', 'VeryRough', '0.0742', '0.2', '235', '1970', '0.9', '0.1',
                    '0.5'],
        "m182q27": ['Material', 'BioPCM M182/Q27', 'VeryRough', '0.0742', '0.2', '235', '1970', '0.9', '0.1',
                    '0.5']}

    materials_enthalpy_lookup = {
        "m182q29": ['MaterialProperty:PhaseChange', 'BioPCM M182/Q29', '0', '-20', '1', '0', '5', '5', '9850',
                    '10',
                    '19701', '15', '29552', '25', '54185', '26', '89364', '27', '162498', '28', '162498', '29',
                    '260685',
                    '30', '263729', '31', '267580', '35', '272472', '45', '295887', '50', '308500', '100', '322093'],
        "m182q29concentrated": ['MaterialProperty:PhaseChange', 'BioPCM M182/Q29 concentrated', '0', '-20', '1',
                                '0', '5', '5', '9850', '10', '19701', '15', '29552', '25', '54185', '26', '89364',
                                '27',
                                '162498', '28', '162498', '29', '260685', '30', '263729', '31', '267580', '35',
                                '272472',
                                '45', '295887', '50', '308500', '100', '322093'],
        "m182q23": ['MaterialProperty:PhaseChange', 'BioPCM M182/Q23', '0', '-20', '1', '0', '12', '10', '23058',
                    '15', '32580', '20', '41280', '21.5', '55230', '22', '81820', '22.5', '128509', '23', '201879',
                    '24',
                    '236860', '25', '245462', '27', '249194', '30', '254503', '35', '258813', '45', '267178', '100',
                    '300420'],
        "m182q25": ['MaterialProperty:PhaseChange', 'BioPCM M182/Q25', '0', '-20', '1', '0', '8', '10', '19290',
                    '15', '27240', '20', '36990', '23', '42867', '24', '56221', '24.5', '83245', '25', '133649', '25.5',
                    '201879', '26', '236860', '28', '247994', '32', '254449', '35', '257761', '45', '266724', '100',
                    '322285'],
        "m182q27": ['MaterialProperty:PhaseChange', 'BioPCM M182/Q27', '0', '-20', '1', '0', '5', '10', '16458',
                    '15', '23562', '20', '32561', '25', '43078', '26', '57014', '26.5', '84146', '27', '134578', '27.5',
                    '202864', '28', '237015', '30', '251278', '32', '255234', '35', '258320', '45', '267324', '100',
                    '322093']}

    def __init__(self, name, idf_input=[], output_folder='', weather_file=''):
        self.name = name
        self.idf_input = idf_input
        self.max_thick = 0.23
        self.min_thick = 0.02
        self.output_folder = output_folder
        self.weather_file = weather_file

    def write_idf_file(self, idf_object, file_path):
        f = open(file_path, "w")
        keys = idf_object.keys()
        for key in keys:
            f.write('!-   ===========  ALL OBJECTS IN CLASS: ' + key + ' ===========\n')
            f.write('\n')
            for item in idf_object[key]:
                f.write(''.join([x + ',\n' for x in item[:-1]]) + item[-1] + ';\n')
            f.write('\n')

        f.close()

        # idf = IDF(file_path)
        # idf.saveas(file_path)

    def set_building_orientation(self, idf_object, degree):
        idf_object['BUILDING'][0][2] = degree
        return idf_object

    def set_output_variables(self, idf_object):
        # idf_object['OUTPUT:TABLE:SUMMARYREPORTS'] = []
        # idf_object['OUTPUTCONTROL:TABLE:STYLE'] = []
        # idf_object['OUTPUT:ENVIRONMENTALIMPACTFACTORS'] = []
        # idf_object['OUTPUT:METER'] = []
        # idf_object['OUTPUTCONTROL:ILLUMINANCEMAP:STYLE'] = []
        # idf_object['OUTPUT:SURFACES:DRAWING'] = []
        # idf_object['OUTPUT:SURFACES:LIST'] = []
        # idf_object['OUTPUT:CONSTRUCTIONS'] = []
        # idf_object['OUTPUT:VARIABLEDICTIONARY'] = []
        # idf_object['OUTPUT:ENERGYMANAGEMENTSYSTEM'] = []
        # idf_object['OUTPUTCONTROL:REPORTINGTOLERANCES'] = []
        # idf_object['OUTPUT:DIAGNOSTICS'] = []
        idf_object['OUTPUT:VARIABLE'] = []
        idf_object['OUTPUT:VARIABLE'].append(['Output:Variable',
                                              '*',
                                              'Facility Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time',
                                              'Monthly'])
        idf_object['OUTPUT:VARIABLE'].append(['Output:Variable',
                                              '*',
                                              'Facility Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time',
                                              'Annual'])
        idf_object['OUTPUT:VARIABLE'].append(['Output:Variable',
                                              '*',
                                              'District Cooling Chilled Water Energy',
                                              'Monthly'])
        idf_object['OUTPUT:VARIABLE'].append(['Output:Variable',
                                              '*',
                                              'District Cooling Chilled Water Energy',
                                              'Annual'])

        idf_object['OUTPUT:METER'].append(['Output:Meter',
                                           'DistrictHeating:Facility',
                                           'Monthly'])
        idf_object['OUTPUT:METER'].append(['Output:Meter',
                                           'DistrictHeating:Facility',
                                           'Annual'])

        idf_object['OUTPUT:METER'].append(['Output:Meter',
                                           'DistrictCooling:Facility',
                                           'Monthly'])
        idf_object['OUTPUT:METER'].append(['Output:Meter',
                                           'DistrictCooling:Facility',
                                           'Annual'])
        idf_object['ENVIRONMENTALIMPACTFACTORS'][0][1] = '3.0'
        return idf_object

    def add_new_material(self, idf_object, material_enthalpy, material_props, suffix):
        material_e_copy = material_enthalpy.copy()
        material_p_copy = material_props.copy()
        new_name = material_e_copy[1] + ('-' + suffix if suffix != '' else '')
        material_e_copy[1] = new_name
        material_p_copy[1] = new_name
        idf_object['MATERIALPROPERTY:PHASECHANGE'].append(material_e_copy)
        idf_object['MATERIAL'].append(material_p_copy)

        return idf_object

    def set_material_to_constructions(self, idf_object, material_name):
        for i, construction in enumerate(idf_object['CONSTRUCTION']):
            for j, detail in enumerate(construction):
                if "BioPCM" in detail:
                    idf_object['CONSTRUCTION'][i][j] = material_name
        return idf_object

    def set_material_thickness(self, idf_object, thickness, material_name):
        for i, material in enumerate(idf_object['MATERIAL']):
            if material[1] == material_name:
                idf_object['MATERIAL'][i][3] = thickness
        return idf_object

    def generateIDFs(self):

        # MAKE NEW DIRECTORY AND copy files to output folder
        run_folder = self.output_folder + "/" + self.name
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)
        for file_key, file in self.idf_input.items():
            new_path = run_folder + "/" + file_key + ".idf"
            copy(file, new_path)
            self.idf_input[file_key] = new_path

        # Read IDF file and generate all needed combinations
        for file_key, file in self.idf_input.items():
            with open(file, 'r', encoding="utf8", errors='ignore') as f:
                idf = parseidf.parse(f.read())

            # Add output variables
            idf = self.set_output_variables(idf)

            # Remove existing PCM materials
            idf['MATERIAL'] = [item for item in idf['MATERIAL'] if ("BioPCM" not in item[1])]
            idf['MATERIALPROPERTY:PHASECHANGE'] = [item for item in idf['MATERIALPROPERTY:PHASECHANGE'] if
                                                   ("BioPCM" not in item[1])]

            # Add materials to the IDF (lookups)
            for key in self.materials_enthalpy_lookup.keys():
                idf = self.add_new_material(idf, self.materials_enthalpy_lookup[key], self.materials_props_lookup[key],
                                            '')

            for key, material in self.materials_props_lookup.items():
                # Set material to construction
                self.set_material_to_constructions(idf, material[1])

                # Loop on all thicknesses
                for thickness in np.arange(float(self.min_thick), float(self.max_thick), 0.05):
                    self.set_material_thickness(idf, str(thickness), material[1])
                    str_thickness = "{:.2f}".format(thickness)
                    filename = file_key + "-" + key + "-" + str_thickness
                    idf_folder = run_folder + "/" + filename
                    os.mkdir(idf_folder)
                    file_path = idf_folder + "/" + filename + ".idf"
                    self.write_idf_file(idf, file_path)

    def generateFiles(self, iterate_on, allocation, material, thickness):
        allocations = ['innermost', 'inside', 'outermost', 'outerside']
        thicknesses = ['0.02', '0.07', '0.12', '0.17', '0.22']
        materials = ['m182q23', 'm182q25', 'm182q27', 'm182q29', 'm182q29concentrated']
        files = []
        if iterate_on == 'allocation':
            for item in allocations:
                files.append(item + '-' + material + '-' + thickness)
        elif iterate_on == 'material':
            for item in materials:
                files.append(allocation + '-' + item + '-' + thickness)
        elif iterate_on == 'thickness':
            for item in thicknesses:
                files.append(allocation + '-' + material + '-' + item)

        return files

    def runEnergyPlus(self, outout_folder, model_name):
        model_folder = outout_folder + "/" + model_name
        file_path = model_folder + "/" + model_name + ".idf"
        eso_found = False
        for subdir, dirs, files in os.walk(model_folder):
            files = [fi for fi in files if fi.endswith(".eso")]
            for file in files:
                if file == 'eplusout.eso':
                    eso_found = True

        if eso_found == True:
            print('Skipping ' + model_name)
        else:
            print('Simulating ' + model_name)
            cmd = '/Applications/EnergyPlus-9-1-0/energyplus-9.1.0 -w ' + self.weather_file + ' -d ' + model_folder + ' -r ' + file_path
            print(cmd)
            os.system(cmd)
            time.sleep(120)

    def evaluateOutputs(self, output_folder, model_name):
        model_folder = output_folder + "/" + model_name
        file_path = model_folder + "/eplusout.eso"
        dd, data = esoreader.read(file_path)
        output_data = {}
        thermal_variables = dd.find_variable(
            'Facility Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time')
        for thermal_variable in thermal_variables:
            frequency, key, variable = thermal_variable
            if frequency == "Annual":
                idx = dd.index[frequency, key, variable]
                time_series = data[idx]
                output_data['thermal_comfort'] = sum(time_series)

        thermal_variables = dd.find_variable('DistrictHeating:Facility')
        for thermal_variable in thermal_variables:
            frequency, key, variable = thermal_variable
            if frequency == "Annual":
                idx = dd.index[frequency, key, variable]
                time_series = data[idx]
                output_data['heating'] = sum(time_series) / 3600000

        thermal_variables = dd.find_variable(
            'DistrictCooling:Facility')
        for thermal_variable in thermal_variables:
            frequency, key, variable = thermal_variable
            if frequency == "Annual":
                idx = dd.index[frequency, key, variable]
                time_series = data[idx]
                output_data['cooling'] = sum(time_series) / 3600000

        parts = model_name.split('-')
        output_data['total_energy'] = output_data['cooling'] + output_data['heating']
        output_data['initial_cost'] = 2678 * self.material_cost[parts[1]] * float(parts[2])
        output_data['running_cost'] = (output_data['cooling'] + output_data['heating']) * 1.6

        return output_data

    def getSuccessfulModels(self, outputs):
        models = []
        for model_name in list(outputs):
            output_data = outputs[model_name]
            if output_data['thermal_comfort'] > 300:
                del outputs[model_name]

        minimum_energy = -1
        temp_model = ''
        for model_name in list(outputs):
            output_data = outputs[model_name]
            if minimum_energy == -1 or output_data['total_energy'] < minimum_energy:
                minimum_energy = output_data['total_energy']
                temp_model = model_name
        models.append(temp_model)
        del outputs[temp_model]

        minimum_cost = -1
        temp_model = ''
        for model_name in list(outputs):
            output_data = outputs[model_name]
            if minimum_cost == -1 or output_data['initial_cost'] < minimum_cost or (
                    output_data['initial_cost'] == minimum_cost and output_data['total_energy'] < outputs[temp_model]['total_energy']):
                minimum_cost = output_data['initial_cost']
                temp_model = model_name
        models.append(temp_model)
        del outputs[temp_model]

        return models

    def simulate(self):

        outout_folder = self.output_folder + "/" + self.name

        outputs = {}
        all_outputs = {}
        # First run on innermost with thickness = 0.0007 on different materials
        model_names = self.generateFiles('material', 'innermost', '', '0.07')
        for model_name in model_names:
            self.runEnergyPlus(outout_folder, model_name)
            outputs[model_name] = self.evaluateOutputs(outout_folder, model_name)
            all_outputs[model_name] = outputs[model_name]

        passed_models = self.getSuccessfulModels(outputs)
        passed_models_thickness = []

        for passed_model in passed_models:
            parts = passed_model.split('-')
            model_names = self.generateFiles('thickness', 'innermost', parts[1], '')
            outputs = {}
            for model_name in model_names:
                self.runEnergyPlus(outout_folder, model_name)
                outputs[model_name] = self.evaluateOutputs(outout_folder, model_name)
                all_outputs[model_name] = outputs[model_name]

            print(outputs)
            passed_models_thickness += self.getSuccessfulModels(outputs)

        passed_models_allocation = []
        print(passed_models_thickness)
        all_outputs = {}

        for passed_model in passed_models_thickness:
            parts = passed_model.split('-')
            model_names = self.generateFiles('allocation', '', parts[1], parts[2])
            outputs = {}
            for model_name in model_names:
                self.runEnergyPlus(outout_folder, model_name)
                outputs[model_name] = self.evaluateOutputs(outout_folder, model_name)
                all_outputs[model_name] = outputs[model_name]
            passed_models_allocation += self.getSuccessfulModels(outputs)
        self.visualize(all_outputs)
        return
        final_outputs = {}
        for passed_model in passed_models_allocation:
            final_outputs[passed_model] = self.evaluateOutputs(outout_folder, passed_model)

        self.visualize(all_outputs)
        return
    def visualize(self, outputs):
        plot_data = {
            'name': [],
            'material': [],
            'allocation': [],
            'thickness': [],
            'color': [],
            'heating': [],
            'cooling': [],
            'total_energy': [],
            'initial_cost': [],
            'running_cost': [],
            'thermal_comfort': [],
        }
        count = 1
        color_continuous_scale = []
        total_files = len(outputs)
        ticktext = []
        for model_name in outputs:
            data = outputs[model_name]
            parts = model_name.split('-')
            plot_data['material'].append(parts[1])
            plot_data['allocation'].append(parts[0])
            plot_data['thickness'].append(parts[2])
            plot_data['color'].append(count)
            plot_data['name'].append(model_name)
            plot_data['total_energy'].append(data['total_energy'])
            plot_data['initial_cost'].append(data['initial_cost'])
            plot_data['running_cost'].append(data['running_cost'])
            plot_data['heating'].append(data['heating'])
            plot_data['cooling'].append(data['cooling'])
            plot_data['thermal_comfort'].append(data['thermal_comfort'])

            start = float((count - 1)) / float(total_files)
            end = start + (1 / (float(total_files)))
            color_continuous_scale.append([float("{:.2f}".format(start)), self.colors[count]])
            color_continuous_scale.append([float("{:.2f}".format(end)), self.colors[count]])
            ticktext.append(model_name)
            count += 1

        # Create DataFrame
        df = pd.DataFrame(plot_data)

        # Print the output.
        fig = px.parallel_coordinates(df,
                                      dimensions=[
                                          "name",
                                          "heating",
                                          "cooling",
                                          'total_energy',
                                          'initial_cost',
                                          'running_cost',
                                          "thermal_comfort",

                                      ],
                                      # title="Optimization Process",
                                      color='color',
                                      color_continuous_scale=color_continuous_scale
                                      )
        fig.update_layout(coloraxis_showscale=True)
        fig.update_layout(coloraxis_colorbar=dict(
            title="",
            tickvals=list(range(1, len(outputs)+1)),
            ticktext=ticktext,
            lenmode="pixels", len=300,
        ))
        webbrowser.open('http://localhost:8050', new=2)
        app = dash.Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig),
        ])
        app.run_server(debug=False, host='localhost', port=8050, use_reloader=False)

        fig.show(config={
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'height': 500,
                'width': 1000,
                'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
            }})

# Calculate outputs

# Form CSV

# Plot output
