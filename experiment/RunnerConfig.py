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

from datetime import datetime
import subprocess
import signal
import time
import shlex
import os
import pandas as pd
from dotenv import load_dotenv, dotenv_values
import random
import psutil
import re

@CodecarbonWrapper.emission_tracker(
    data_columns=[CCDataCols.ENERGY_CONSUMED],
    country_iso_code="NLD" # your country code
)


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    timestamp_start  = None
    timestamp_end  = None
    response = None

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
    time_between_runs_in_ms:    int             = 90000

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
        factor1 = FactorModel("model", ['llama3.1:8b', 'gemma:2b', 'gemma:7b', 'phi3:3.8b', 'qwen2:1.5b', 'qwen2:7b', 'mistral:7b'])
        factor2 = FactorModel("method", ['remote', 'local'])
        factor3 = FactorModel("length", ['100','500','1000'])
        self.run_table_model = RunTableModel(
            factors=[factor1, factor2, factor3],
            data_columns=['topic', 'execution_time', '%cpu_usage','%gpu_usage', '%memory_usage'],
            shuffle=True,
            repetitions = 30,
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

        model = context.run_variation["model"]
        method = context.run_variation["method"]
        size = context.run_variation["length"]

        topic_list_data = pd.read_csv(Path.cwd()/"experiment/topics.csv")
        topics = topic_list_data["Topic"].tolist()

        self.topic = random.choice(topics)

        prompt = f"In {size} words, please give me information about " + self.topic

        if method == "local":
            url = "localhost"
        else:
            load_dotenv()
            url = os.getenv("SERVER_IP")

        data = "'{" + f'"model": "{model}", "prompt": "{prompt}", "stream": false' + "}'"

        cmd = f'curl http://{url}:11434/api/generate -d {data}'
        self.target = subprocess.Popen(shlex.split(cmd))

        output.console_log("Config.start_run() called!")

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        output.console_log("Config.start_measurement() called!")

        # Command to start power metrics (e.g., GPU power measurement)
        powermetrics_cmd = f'sudo powermetrics -o {context.run_dir/"powermetrics.txt"} -i 100 --samplers gpu_power'

        # Start the GPU profiler (optional part depending on system and available sensors)
        self.gpu_profiler = subprocess.Popen(shlex.split(powermetrics_cmd))

        # Create CSV file to store CPU and memory usage data
        cpu_mem_usage_file = context.run_dir / 'cpu_mem_usage.csv'

        with open(cpu_mem_usage_file, mode='w') as file:
            # CSV header
            file.write("timestamp,cpu_usage,memory_usage\n")

        # Monitor the target subprocess (curl command)
        psdf = pd.DataFrame(columns=['cpu_usage', 'memory_usage'])  # Remove if you don't want to keep it in memory

        process = psutil.Process(self.target.pid)  # Track the curl process
        while psutil.pid_exists(self.target.pid):
            try:
                # Get CPU and memory usage stats
                cpu_usage_percentage = psutil.cpu_percent(interval=0.1)  # 0.1s interval
                memory_usage_percentage = psutil.virtual_memory().percent
                process_memory_usage_percentage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Log the data to the CSV file
                with open(cpu_mem_usage_file, mode='a') as file:
                    file.write(f"{timestamp},{cpu_usage_percentage},{memory_usage_percentage}\n")

                # Also store in the DataFrame (if needed for future use)
                psdf.loc[len(psdf.index)] = [cpu_usage_percentage, memory_usage_percentage]

                time.sleep(1)  # Sleep for 1 second to avoid over-polling the system
            except psutil.NoSuchProcess:
                # Handle if the process has ended prematurely
                break

        self.performance_profiler_df = psdf  # Save DataFrame for later use
        output.console_log("Measurement completed and saved.")

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        output.console_log("Config.interact() called!")

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        os.kill(self.target.pid, signal.SIGKILL)
        os.kill(self.gpu_profiler.pid, signal.SIGKILL)
        self.gpu_profiler.kill()
        self.target.kill()

        output.console_log("Config.stop_measurement called!")

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        self.timestamp_end = datetime.now()
        output.console_log("Config.stop_run() called!")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""

        output.console_log("Config.populate_run_data() called!")

        pattern = re.compile(
            r'GPU HW active residency:\s+(?P<gpu_hw_active_residency>\d+\.\d+)%\s+\(.*?\)\s+'
        )

        gpu_hw_active_residency_values = []  # To store the residency values for median calculation
        gpu_percentage_mean = 0

        # Open and read the file
        with open(context.run_dir/"powermetrics.txt", mode='r') as file:
            content = file.read()

            # Extracting GPU data from the content
            matches = pattern.finditer(content)
            for match in matches:
                # Convert the GPU HW Active Residency to float and store it
                gpu_hw_active_residency_values.append(float(match.group('gpu_hw_active_residency')))

        # Calculate and return the median
        if gpu_hw_active_residency_values:
            gpu_percentage_mean = sum(gpu_hw_active_residency_values) / len(gpu_hw_active_residency_values)


        run_data = {
            'topic': self.topic,
            'execution_time': (self.timestamp_end - self.timestamp_start).total_seconds(),
            '%cpu_usage': round(self.performance_profiler_df['cpu_usage'].mean(), 3),
            '%gpu_usage': round(gpu_percentage_mean, 3),
            '%memory_usage': round(self.performance_profiler_df['memory_usage'].mean(), 3),
        }
        os.remove(context.run_dir/"powermetrics.txt") # Remove to save storage space since each file cost approx. 150mb
        return run_data

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""

        output.console_log("Config.after_experiment() called!")
        # Define the path to the run table CSV
        run_table_path = self.results_output_path / self.name / 'run_table.csv'

        # Load the existing run table CSV into a DataFrame
        try:
            run_table_df = pd.read_csv(run_table_path)
        # Check if the column 'codecarbon__energy_consumed' exists
            if 'codecarbon__energy_consumed' in run_table_df.columns:
                # Convert energy consumed from kWh to Joules and create a new column
                run_table_df['energy_consumed(J)'] = run_table_df['codecarbon__energy_consumed'] * 3_600_000

                # Optionally, you may want to round the values to a specific number of decimal places
                run_table_df['energy_consumed(J)'] = run_table_df['energy_consumed(J)'].round(3)

                # Save the updated DataFrame back to CSV
                run_table_df.to_csv(run_table_path, index=False)
                output.console_log("Added 'energy_consumed(J)' column to run_table.csv successfully.")
            else:
                output.console_log("'codecarbon__energy_consumed' column not found in run_table.csv.")
        except FileNotFoundError:
            output.console_log(f"Run table CSV not found at {run_table_path}.")
        except Exception as e:
            output.console_log(f"An error occurred while updating run_table.csv: {e}")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
