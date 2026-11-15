import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment import ROOT, file_sha256, write_json
from experiment_metrics import fit_temperature, metrics

DATASETS = ['dtd', 'oxford_flowers']
METHODS = ['maple', 'promptsrc', 'tcpt']
NAMES = dict(maple='MaPLe', promptsrc='PromptSRC', tcpt='MaPLe+TCPT')
DATASET_NAMES = dict(dtd='DTD', oxford_flowers='Flowers102')
COLORS = dict(maple='#58718D', promptsrc='#258C7C', tcpt='#C36B3B')
MEASURES = ['base_accuracy', 'new_accuracy', 'h', 'base_ece', 'new_ece', 'base_nll', 'new_nll']
OUT = ROOT / 'outputs' / 'B'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'


def write_csv(name, rows):
    assert rows
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['online_offline_check']
    assert file_sha256(directory / 'predictions.npz') == metadata['predictions_sha256']
    with np.load(directory / 'predictions.npz', allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    return data, metadata


def run_measures(base, new):
    denominator = base['accuracy'] + new['accuracy']
    harmonic = 2 * base['accuracy'] * new['accuracy'] / denominator if denominator else 0.0
    return dict(base_accuracy=base['accuracy'], new_accuracy=new['accuracy'], h=harmonic,
                base_ece=base['ece'], new_ece=new['ece'], base_nll=base['nll'], new_nll=new['nll'])


def summarize(rows):
    summary = []
    for dataset in DATASETS:
        for method in METHODS:
            selected = [row for row in rows if row['dataset'] == dataset and row['method'] == method]
            assert sorted(row['seed'] for row in selected) == [1, 2, 3]
            row = dict(dataset=dataset, method=method, seeds=3)
            for measure in MEASURES:
                values = [entry[measure] for entry in selected]
                row[measure + '_mean'] = float(np.mean(values))
                row[measure + '_std'] = float(np.std(values, ddof=1))
            summary.append(row)
    return summary


def save_figure(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / (name + '.png'), dpi=180, bbox_inches='tight')
    figure.savefig(FIGURES / (name + '.pdf'), bbox_inches='tight')
    plt.close(figure)


def plots(raw_summary, scaled_summary, reliability):
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    raw_index = {(row['dataset'], row['method']): row for row in raw_summary}
    scaled_index = {(row['dataset'], row['method']): row for row in scaled_summary}
    x = np.arange(len(METHODS))
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column, (measure, label) in enumerate([('base_accuracy', 'Base'), ('new_accuracy', 'New'), ('h', 'H')]):
            axis = axes[row_index, column]
            values = [raw_index[dataset, method][measure + '_mean'] for method in METHODS]
            errors = [raw_index[dataset, method][measure + '_std'] for method in METHODS]
            axis.bar(x, values, yerr=errors, capsize=4, color=[COLORS[method] for method in METHODS], width=0.62)
            axis.set_xticks(x, [NAMES[method] for method in METHODS], rotation=12)
            axis.set_ylim(0, 100)
            axis.set_ylabel('Accuracy / H (%)')
            axis.set_title(DATASET_NAMES[dataset] + ' — ' + label)
            axis.grid(axis='y', alpha=0.18)
            axis.set_axisbelow(True)
    figure.suptitle('16-shot Base-to-New recognition, mean and sample SD over three seeds')
    save_figure(figure, 'recognition_base_new_h')

    figure, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column, (measure, label) in enumerate([('new_ece', 'New ECE (%)'), ('new_nll', 'New NLL (nats)')]):
            axis = axes[row_index, column]
            for offset, index, color, legend in [(-0.18, raw_index, '#58718D', 'Raw'),
                                                   (0.18, scaled_index, '#258C7C', 'Temperature scaled')]:
                values = [index[dataset, method][measure + '_mean'] for method in METHODS]
                errors = [index[dataset, method][measure + '_std'] for method in METHODS]
                axis.bar(x + offset, values, yerr=errors, capsize=3, width=0.34, color=color, label=legend)
            axis.set_xticks(x, [NAMES[method] for method in METHODS])
            axis.set_ylabel(label)
            axis.set_title(DATASET_NAMES[dataset])
            axis.set_ylim(bottom=0)
            axis.grid(axis='y', alpha=0.18)
            axis.set_axisbelow(True)
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle('Temperature fitted on Base validation, transferred to New test, three seeds')
    save_figure(figure, 'new_calibration_raw_vs_temperature')

    for version, title in [('raw', 'Raw predictions'), ('temperature', 'Base-validation temperature scaling')]:
        figure = plt.figure(figsize=(12, 10), layout='constrained')
        outer = figure.add_gridspec(2, 2)
        for row_index, dataset in enumerate(DATASETS):
            for column, group in enumerate(['base', 'new']):
                inner = outer[row_index, column].subgridspec(2, 1, height_ratios=[3, 1])
                axis = figure.add_subplot(inner[0])
                histogram = figure.add_subplot(inner[1])
                axis.plot([0, 1], [0, 1], color='#A4A4A4', linewidth=1, linestyle='--')
                for method_index, method in enumerate(METHODS):
                    result = reliability[dataset, method, group, version]
                    counts = np.asarray(result['counts'])
                    valid = counts > 0
                    axis.plot(np.asarray(result['bin_confidence'])[valid], np.asarray(result['bin_accuracy'])[valid],
                              color=COLORS[method], marker='o', markersize=4, label=NAMES[method])
                    centers = (np.arange(10) + 0.5) / 10 + (method_index - 1) * 0.026
                    histogram.bar(centers, counts, width=0.024, color=COLORS[method])
                axis.set(xlim=(0, 1), ylim=(0, 1), ylabel='Empirical accuracy',
                         title=f'{DATASET_NAMES[dataset]} — {group.capitalize()}')
                axis.grid(alpha=0.15)
                axis.legend(frameon=False, fontsize=8, loc='upper left')
                histogram.set(xlim=(0, 1), xlabel='Confidence', ylabel='Samples')
                histogram.set_xticks(np.linspace(0, 1, 6))
        figure.suptitle('Seed 1 reliability and bin sample counts — ' + title)
        save_figure(figure, 'reliability_seed1_' + version)


def main():
    progress = json.loads((ROOT / 'records' / 'B_progress.json').read_text())
    assert progress['completed'] == 18 and len(progress['clip_references']) == 4
    raw_rows, temperature_rows, scaled_rows, sensitivity, costs, reliability_rows = [], [], [], [], [], []
    reliability = {}
    temperatures = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in [1, 2, 3]:
                run = Path(progress['selected_runs'][f'{dataset}/{method}/seed{seed}'])
                status = json.loads((run / 'run_status.json').read_text())
                assert status['status'] == 'complete' and status['cross_process_probe_exact']
                data = {}
                metadata = {}
                for group in ['test_base', 'test_new', 'val_base']:
                    data[group], metadata[group] = load_predictions(run / 'predictions' / group)
                started = time.perf_counter()
                temperature = fit_temperature(data['val_base']['logits'], data['val_base']['labels'])
                fitting_seconds = time.perf_counter() - started
                temperature.update(dataset=dataset, method=method, seed=seed, run_id=status['run_id'],
                                   validation_prediction_sha256=metadata['val_base']['predictions_sha256'],
                                   validation_config_sha256=metadata['val_base']['config_sha256'],
                                   validation_source='val_base', fitting_seconds=fitting_seconds)
                raw, scaled = {}, {}
                top1_checks = {}
                for group in ['base', 'new']:
                    prediction = data['test_' + group]
                    logits, labels = prediction['logits'], prediction['labels']
                    raw[group] = metrics(logits, labels)
                    scaled[group] = metrics(logits, labels, temperature=temperature['temperature'])
                    top1_checks[group] = bool(np.array_equal(logits.argmax(axis=1),
                                             (logits.astype(np.float64) / temperature['temperature']).argmax(axis=1)))
                    assert top1_checks[group]
                    assert raw[group]['accuracy'] == scaled[group]['accuracy']
                    for bins in [10, 15, 20]:
                        for version, value in [('raw', 1.0), ('temperature', temperature['temperature'])]:
                            result = metrics(logits, labels, bins=bins, temperature=value)
                            sensitivity.append(dict(dataset=dataset, method=method, seed=seed, group=group,
                                                    version=version, bins=bins, ece=result['ece']))
                            if seed == 1 and bins == 10:
                                reliability[dataset, method, group, version] = result
                                for index in range(10):
                                    reliability_rows.append(dict(dataset=dataset, method=method, seed=seed, group=group,
                                        version=version, bin=index + 1, lower=index / 10, upper=(index + 1) / 10,
                                        count=result['counts'][index], accuracy=result['bin_accuracy'][index],
                                        confidence=result['bin_confidence'][index]))
                temperature.update(top1_preserved=top1_checks, test_raw=raw, test_scaled=scaled)
                write_json(run / 'temperature.json', temperature)
                temperatures.append(temperature)
                resources = json.loads((run / 'training_resources.json').read_text())
                key = dict(dataset=dataset, method=method, seed=seed, run_id=status['run_id'])
                raw_rows.append(dict(**key, **run_measures(raw['base'], raw['new']),
                                     training_seconds=resources['training_seconds'],
                                     initialization_seconds=resources['initialization_seconds'],
                                     inference_seconds=sum(value['evaluation_seconds'] for value in metadata.values()),
                                     peak_memory_gib=resources['peak_memory_gib']))
                scaled_row = dict(**key, **run_measures(scaled['base'], scaled['new']))
                scaled_rows.append(scaled_row)
                temperature_rows.append(dict(**key, temperature=temperature['temperature'],
                    validation_nll_before=temperature['validation_nll_before'],
                    validation_nll_after=temperature['validation_nll_after'], validation_n=temperature['validation_n'],
                    boundary_hit=temperature['boundary_hit'], base_top1_preserved=top1_checks['base'],
                    new_top1_preserved=top1_checks['new'], fitting_seconds=fitting_seconds,
                    **run_measures(scaled['base'], scaled['new'])))
                costs.append(dict(**key, training_seconds=resources['training_seconds'],
                    initialization_seconds=resources['initialization_seconds'],
                    checkpoint_check_seconds=resources['checkpoint_probe_seconds'],
                    test_base_seconds=metadata['test_base']['evaluation_seconds'],
                    test_new_seconds=metadata['test_new']['evaluation_seconds'],
                    val_base_seconds=metadata['val_base']['evaluation_seconds'], temperature_seconds=fitting_seconds,
                    peak_training_memory_gib=resources['peak_memory_gib'],
                    peak_evaluation_memory_gib=max(value['peak_memory_gib'] for value in metadata.values())))
    raw_summary, scaled_summary = summarize(raw_rows), summarize(scaled_rows)
    for scaled_row, raw_row in zip(scaled_summary, raw_summary):
        for measure in ['base_ece', 'new_ece', 'base_nll', 'new_nll']:
            scaled_row[measure + '_delta_mean'] = scaled_row[measure + '_mean'] - raw_row[measure + '_mean']
    paired = []
    for dataset in DATASETS:
        for seed in [1, 2, 3]:
            row = dict(dataset=dataset, seed=seed)
            for version, values in [('raw', raw_rows), ('temperature', scaled_rows)]:
                subset = {entry['method']: entry for entry in values if entry['dataset'] == dataset and entry['seed'] == seed}
                for measure in MEASURES:
                    row[version + '_delta_' + measure] = subset['tcpt'][measure] - subset['maple'][measure]
            paired.append(row)
    references = []
    for key, directory in progress['clip_references'].items():
        dataset, group = key.split('/')
        prediction, metadata = load_predictions(Path(directory))
        result = metrics(prediction['logits'], prediction['labels'])
        references.append(dict(dataset=dataset, candidate_group=group, template=metadata['text_template'],
                               n=result['n'], classes=result['classes'], accuracy=result['accuracy'],
                               ece=result['ece'], nll=result['nll']))
    for filename, rows in [('runs_raw.csv', raw_rows), ('summary_raw.csv', raw_summary),
                           ('temperature_runs.csv', temperature_rows), ('summary_temperature.csv', scaled_summary),
                           ('paired_tcpt_vs_maple.csv', paired), ('clip_reference.csv', references),
                           ('ece_bins_sensitivity.csv', sensitivity), ('reliability_seed1.csv', reliability_rows),
                           ('costs.csv', costs)]:
        write_csv(filename, rows)
    plots(raw_summary, scaled_summary, reliability)
    report = dict(passed=True, completed_at=datetime.now().astimezone().isoformat(), formal_runs=18,
                  raw_test_predictions=36, base_validation_predictions=18, temperatures=18, clip_references=4,
                  seed_count=3, temperature_top1_checks_passed=True,
                  temperature_boundary_hits=sum(record['boundary_hit'] for record in temperatures),
                  summary_raw=raw_summary, summary_temperature=scaled_summary,
                  training_hours=sum(row['training_seconds'] for row in costs) / 3600,
                  configuration_sha256=file_sha256(ROOT / 'records' / 'config_manifest.json'))
    write_json(ROOT / 'records' / 'analysis_checks.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
