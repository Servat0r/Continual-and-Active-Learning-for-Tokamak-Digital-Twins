from typing import *
import sys
from src.database import *
from src.utils.datasets import *


# EXAMPLE USAGE AND DEMONSTRATION
if __name__ == "__main__":

    print("Initializing ...")

    # Initialize the secure database
    db = SecureMLExperimentDB("sqlite:///" + DEFAULT_DB_TEST_FILE, echo=False, overwrite_db=True, overwrite_consent=False)
    
    print("=== Secure SQLAlchemy ORM ML Experiment Database Demo ===\n")
    
    try:
        # 1. CREATE OPERATIONS WITH UNIQUE NAME HANDLING
        print("1. Creating sample configurations with unique name handling...")
        
        # Create configurations
        general = General(
            mode="cl",
            num_campaigns=10,
            train_mb_size=4096,
            eval_mb_size=1024,
            train_epochs=200
        )
        general_id = db.create_record(general)
        print(f"Created General config with ID: {general_id}")
        
        scenario = Scenario(
            simulator_type="qualikiz",
            pow_type="mixed",
            cluster_type="wmhd_based",
            dataset_type="not_null",
            task="regression",
            input_columns=QUALIKIZ_MIXED_INPUTS,
            output_columns=QUALIKIZ_MIXED_OUTPUTS,
            normalize_inputs=True,
            normalize_outputs=False,
            normalization_type="first-exp"
        )
        scenario_id = db.create_record(scenario)
        print(f"Created Scenario with ID: {scenario_id}")
        
        architecture = Architecture(
            model_type="MLP",
            model_folder="...",
            parameters={
                "input_size": 15,
                "output_size": 4,
                "hidden_size": 1024,
                "hidden_layers": 2,
                "drop_rate": 0.5
            }
        )
        arch_id = db.create_record(architecture)
        print(f"Created Architecture with ID: {arch_id}")

        print(f"Testing tags updating")
        new_arch = db.add_or_update_tags(Architecture, arch_id, tags={'type': 'std_mlp', 'author': 'Salvatore Correnti'})
        print(new_arch)
        
        loss = Loss(
            name="MSE",
            parameters={"reduction": "mean", "weight": None}
        )
        loss_id = db.create_record(loss)
        print(f"Created Loss with ID: {loss_id}")
        
        optimizer = Optimizer(
            name="AdamW",
            parameters={"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8}
        )
        optimizer_id = db.create_record(optimizer)
        print(f"Created Optimizer with ID: {optimizer_id}")
        
        scheduler = Scheduler.from_dict({
            "name": "ReduceLROnPlateau",
            "parameters": {
                "metric": "train_loss",
                "first_exp_only": False,
                "factor": 0.5,
                "patience": 21,
                "mode": "min",
                "threshold_mode": "abs",
                "threshold": 1.0,
                "min_lr": 1e-4
            }
        })
        scheduler_id = db.create_record(scheduler)
        print(f"Created Scheduler with ID: {scheduler_id}")
        
        strategy = Strategy(
            name="fine_tuning",
            from_scratch=False,
            parameters={"freeze_layers": 10, "unfreeze_epoch": 20}
        )
        strategy_id = db.create_record(strategy)
        print(f"Created Strategy with ID: {strategy_id}")
        
        # Test unique experiment name creation
        print("\n--- Testing Unique Experiment Name Generation ---")

        # Test Reading of records
        with db.get_session() as session:
            schedulers = db.read_records_where(
                Scheduler, {
                    "parameters": {
                        "patience": ('>=', 21),
                        "min_lr": ('<=', 1e-4)
                    }
                },
                as_dict=True
            )
            print(schedulers)

            architectures = db.read_records_where(
                Architecture, {
                    "model_type": "MLP",
                    "parameters": {
                        "hidden_size": 1024
                    }
                },
                as_dict=True
            )
            print(architectures)

        # Create first experiment

        experiment = Experiment(
            id_general=general_id,
            id_scenario=scenario_id,
            id_architecture=arch_id,
            id_loss=loss_id,
            id_optimizer=optimizer_id,
            id_scheduler=scheduler_id,
            id_strategy=strategy_id,
            name="MLP_Experiment",
            num_tasks=4,
            status="invalid"
        )

        exp_id1, exp_name1 = db.create_experiment(experiment)
        print(f"Created Experiment 1: ID={exp_id1}, Name='{exp_name1}'")
        
        # Try to create another with the same name - should auto-generate unique name
        exp_id2, exp_name2 = db.create_experiment(
            Experiment(
                id_general=general_id,
                id_scenario=scenario_id,
                id_architecture=arch_id,
                id_loss=loss_id,
                id_optimizer=optimizer_id,
                id_scheduler=scheduler_id,
                id_strategy=strategy_id,
                name="MLP_Experiment",  # Same name as before
                num_tasks=8,
                status="pending"
            )
        )
        print(f"Created Experiment 2: ID={exp_id2}, Name='{exp_name2}'")
        
        # 3. TEST ADVANCED QUERIES WITH SECURITY
        print("\n--- Testing Secure Advanced Queries ---")
        
        # Read experiments with conditions
        running_experiments = db.read_records_where(Experiment, {"status": "running"})
        print(f"Running experiments: {len(running_experiments)}")
        
        # Test advanced search
        advanced_search_experiments = db.search_experiments_advanced(
            general_conditions={"mode": "CL"},
            scenario_conditions={"cluster_type": ("like", "%mh%")},
            experiment_conditions={"status": ("in", ["running", "init"])},
            limit=10
        )
        print(f"Advanced search experiments: {len(advanced_search_experiments)}")
        #print(advanced_search_experiments)

        # Test Status Updates
        ids = [exp['id'] for exp in advanced_search_experiments]
        db.set_init_to_pending(ids)
        advanced_search_experiments = db.search_experiments_advanced(
            general_conditions={"mode": "CL"},
            scenario_conditions={"cluster_type": ("like", "%mh%")},
            experiment_conditions={"status": "pending"},
            limit=10
        )
        print(f"Advanced search experiments: {len(advanced_search_experiments)}")

        db.set_pending_to_running([exp['id'] for exp in advanced_search_experiments])
        advanced_search_experiments = db.search_experiments_advanced(
            general_conditions={"mode": "CL"},
            scenario_conditions={"cluster_type": ("like", "%mh%")},
            experiment_conditions={"status": "running"},
            limit=10
        )
        print(f"Advanced search experiments: {len(advanced_search_experiments)}")

        db.set_running_to_finished([exp['id'] for exp in advanced_search_experiments])
        advanced_search_experiments = db.search_experiments_advanced(
            general_conditions={"mode": "CL"},
            scenario_conditions={"cluster_type": ("like", "%mh%")},
            experiment_conditions={"status": "finished"},
            limit=10
        )
        print(f"Advanced search experiments: {len(advanced_search_experiments)}")

        # 4. TEST SECURITY FEATURES
        print("\n--- Testing Security Features ---")
        
        try:
            # Test invalid operator (should raise SecurityError)
            db.read_records_where(Experiment, {"status": ("INVALID_OP", "running")})
        except SecurityError as e:
            print(f"✓ Caught expected SecurityError: {e}")
        
        try:
            # Test invalid field name (should raise SecurityError)
            db.read_records_where(Experiment, {"invalid_field": "value"})
        except SecurityError as e:
            print(f"✓ Caught expected SecurityError: {e}")
        
        try:
            # Test oversized IN clause (should raise SecurityError)
            large_list = list(range(200))  # Over the 100 item limit
            db.read_records_where(Experiment, {"id": ("in", large_list)})
        except SecurityError as e:
            print(f"✓ Caught expected SecurityError: {e}")
        
        try:
            # Test invalid experiment name
            db.create_experiment(
                Experiment(
                    id_general=general_id, id_scenario=scenario_id, id_architecture=arch_id,
                    id_loss=loss_id, id_optimizer=optimizer_id, id_scheduler=scheduler_id,
                    id_strategy=strategy_id, name="Invalid/Name<>:", num_tasks=100, status='invalid'
                )
            )
        except ValidationError as e:
            print(f"✓ Caught expected ValidationError: {e}")
        
        # 7. DATABASE STATISTICS AND UTILITIES
        print("\n--- Database Statistics and Utilities ---")
        
        # Get statistics
        stats = db.get_experiment_statistics()
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get database info
        db_info = db.get_database_info()
        print(f"\nDatabase Info:")
        print(f"  URL: {db_info['database_url']}")
        print(f"  Total Records: {db_info['total_records']}")
        print("  Tables:")
        for table_name, info in db_info['tables'].items():
            print(f"    {table_name}: {info['record_count']} records")
