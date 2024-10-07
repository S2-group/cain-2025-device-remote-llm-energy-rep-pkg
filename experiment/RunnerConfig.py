from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

from Plugins.Profilers import CodecarbonWrapper
from Plugins.Profilers.CodecarbonWrapper import DataColumns as CCDataCols

import requests
from datetime import datetime
import subprocess
import signal
import time
import shlex
import os
import pandas as pd
from dotenv import load_dotenv, dotenv_values

@CodecarbonWrapper.emission_tracker(
    data_columns=[CCDataCols.EMISSIONS, CCDataCols.ENERGY_CONSUMED],
    country_iso_code="NLD" # your country code
)


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    timestamp_start  = None
    timestamp_end  = None
    response = None
    powerMetricsProfiler = None

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "new_runner_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path             = ROOT_DIR / 'experiments_output'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 1000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        factor1 = FactorModel("model", ['Llama3.1:8b', 'gemma:2b', 'gemma:7b', 'qwen2:0.5b', 'qwen2:1.5b', 'qwen2:7b', 'mistral:7b'])
        factor2 = FactorModel("method", ['local', 'http'])

        prompts_list_data = pd.read_csv(Path.cwd()/"experiment/prompts.csv")
        prompts = prompts_list_data["Prompt"].tolist()
        output.console_log(prompts)
        factor3 = FactorModel("prompt", prompts)

        factor4 = FactorModel("size", ['100', '500', '1000'])
        self.run_table_model = RunTableModel(
            factors=[factor1, factor2, factor3, factor4],
            data_columns=['response', 'execution_time', 'cpu_usage', 'memory_usage'],
            shuffle=True
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.before_experiment() called!")
        self.timestamp_start = datetime.now()

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""

        output.console_log("Config.before_run() called!")
        self.timestamp_start = datetime.now()


    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        
        output.console_log("Config.start_run() called!")

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""

        model = context.run_variation["model"]
        method = context.run_variation["method"]
        size = context.run_variation["size"]
        prompt = f"In {size} words, please give me information about " + context.run_variation["prompt"]

        if method == "local":
            url = "localhost"
        else:
            load_dotenv()
            url = os.getenv("SERVER_IP")

        data = "'{" + f'"model": "{model}", "prompt": "{prompt}", "stream": false' + "}'"

        output.console_log("Config.start_measurement() called!")
        profiler_cmd = f'sudo energibridge -i 3000 -g -o {context.run_dir / "energibridge.csv"} --summary curl http://{url}:11434/api/generate -d {data}'
        output.console_log(profiler_cmd)
        powermetrics_cmd = f'sudo powermetrics -o {context.run_dir/"powermetrics.txt"} -i 100 --show-usage-summary --hide-cpu-duty-cycle --samplers cpu_power,gpu_power'
        energibridge_log = open(f'{context.run_dir}/energibridge.log', 'w')
        powermetrics_log = open(f'{context.run_dir}/powermetrics.log', 'w')
        self.profiler = subprocess.Popen(shlex.split(profiler_cmd), stdout=energibridge_log)
        self.powerMetricsProfiler = subprocess.Popen(shlex.split(powermetrics_cmd), stdout=powermetrics_log, shell=True)

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        output.console_log("Config.interact() called!")

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        output.console_log("Config.stop_measurement called!")

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        self.timestamp_end = datetime.now()
        self.powerMetricsProfiler.kill()
        output.console_log("Config.stop_run() called!")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""

        output.console_log("Config.populate_run_data() called!")

        df = pd.read_csv(context.run_dir / f"energibridge.csv")
        df["mean_cpu_usage"] = df.loc[:, ["CPU_USAGE_0", "CPU_USAGE_1", "CPU_USAGE_2", "CPU_USAGE_3", "CPU_USAGE_4", "CPU_USAGE_5", "CPU_USAGE_6", "CPU_USAGE_7"]].mean(axis=1);
        run_data = {
            'response': '',
            'execution_time': (self.timestamp_end - self.timestamp_start).total_seconds(),
            'cpu_usage': round(df['mean_cpu_usage'].mean(), 3),
            'memory_usage': round(df['USED_MEMORY'].mean(), 3),
            # 'energy_usage': round(filtered_df['CPU Power'].sum(), 3),
        }
        return run_data

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.after_experiment() called!")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
