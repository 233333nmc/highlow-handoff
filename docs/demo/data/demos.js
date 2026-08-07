window.ROBOTWIN_DEMOS = {
  summary: {
    model: "Gemma4-12B MTP",
    planner_modes: ["completion_status", "vlm_plan_next", "vlm_plan_all"],
    overall_median_ms: 899.3,
    samples: 60
  },
  tasks: [
    {
      name: "place_burger_fries",
      planner_mode: "vlm_plan_next",
      video: "",
      forced_threshold_s: 2.17,
      actual_trigger: "2.20s / frame 55",
      mean_ms: 424.3,
      current_subtask: "Place the burger on the tray.",
      next_subtask: "Place the french fries on the tray.",
      cache_action: "clear stale cache on switch",
      verifier: {
        action: "switch",
        status: "completed",
        raw_output: {
          action: "switch",
          subtask_status: "completed",
          next_subtask: "Place the french fries on the tray.",
          evidence: "The burger is already placed on the tray."
        }
      }
    },
    {
      name: "place_cans_plasticbox",
      planner_mode: "completion_status",
      video: "",
      forced_threshold_s: 2.25,
      actual_trigger: "2.28s / frame 57",
      mean_ms: 828.1,
      current_subtask: "Place the first can in the plastic box.",
      next_subtask: "Place the next can in the plastic box.",
      cache_action: "clear stale cache on switch",
      verifier: {
        action: "switch",
        status: "completed",
        raw_output: {
          action: "switch",
          subtask_status: "completed",
          next_subtask: "Place the next can in the plastic box.",
          evidence: "The target can appears inside the plastic box."
        }
      }
    },
    {
      name: "put_bottles_dustbin",
      planner_mode: "completion_status",
      video: "",
      forced_threshold_s: 4.01,
      actual_trigger: "4.04s / frame 101",
      mean_ms: 1134.2,
      current_subtask: "Put the first bottle into the dustbin.",
      next_subtask: "Put the next bottle into the dustbin.",
      cache_action: "clear stale cache on switch",
      verifier: {
        action: "switch",
        status: "completed",
        raw_output: {
          action: "switch",
          subtask_status: "completed",
          next_subtask: "Put the next bottle into the dustbin.",
          evidence: "The bottle is inside the dustbin region."
        }
      }
    },
    {
      name: "stack_blocks_three",
      planner_mode: "completion_status",
      video: "",
      forced_threshold_s: 2.25,
      actual_trigger: "2.28s / frame 57",
      mean_ms: 1072.2,
      current_subtask: "Stack the first block on the target block.",
      next_subtask: "Stack the next block on the tower.",
      cache_action: "clear stale cache on switch",
      verifier: {
        action: "switch",
        status: "completed",
        raw_output: {
          action: "switch",
          subtask_status: "completed",
          next_subtask: "Stack the next block on the tower.",
          evidence: "The block is aligned on the stack."
        }
      }
    },
    {
      name: "stack_bowls_three",
      planner_mode: "completion_status",
      video: "",
      forced_threshold_s: 4.53,
      actual_trigger: "4.56s / frame 114",
      mean_ms: 938.6,
      current_subtask: "Stack the first bowl on the target bowl.",
      next_subtask: "Stack the next bowl on the bowl stack.",
      cache_action: "clear stale cache on switch",
      verifier: {
        action: "switch",
        status: "completed",
        raw_output: {
          action: "switch",
          subtask_status: "completed",
          next_subtask: "Stack the next bowl on the bowl stack.",
          evidence: "The bowl appears placed on the target bowl."
        }
      }
    },
    {
      name: "place_dual_shoes",
      planner_mode: "completion_status",
      video: "",
      forced_threshold_s: 2.61,
      actual_trigger: "2.64s / frame 66",
      mean_ms: 1041.2,
      current_subtask: "Place the first shoe in the target area.",
      next_subtask: "Place the second shoe in the target area.",
      cache_action: "clear stale cache on switch",
      verifier: {
        action: "switch",
        status: "completed",
        raw_output: {
          action: "switch",
          subtask_status: "completed",
          next_subtask: "Place the second shoe in the target area.",
          evidence: "The first shoe is in the target placement area."
        }
      }
    }
  ],
  results: [
    {
      task: "place_burger_fries",
      forced_threshold_s: 2.17,
      actual_trigger: "2.20s / frame 55",
      mean_ms: 779.5,
      median_ms: 786.7,
      min_ms: 653.8,
      max_ms: 906.2
    },
    {
      task: "place_cans_plasticbox",
      forced_threshold_s: 2.25,
      actual_trigger: "2.28s / frame 57",
      mean_ms: 828.1,
      median_ms: 824.4,
      min_ms: 782.5,
      max_ms: 894.3
    },
    {
      task: "put_bottles_dustbin",
      forced_threshold_s: 4.01,
      actual_trigger: "4.04s / frame 101",
      mean_ms: 1134.2,
      median_ms: 870.7,
      min_ms: 791.7,
      max_ms: 3551.4
    },
    {
      task: "stack_blocks_three",
      forced_threshold_s: 2.25,
      actual_trigger: "2.28s / frame 57",
      mean_ms: 1072.2,
      median_ms: 1083.7,
      min_ms: 914.1,
      max_ms: 1190.8
    },
    {
      task: "stack_bowls_three",
      forced_threshold_s: 4.53,
      actual_trigger: "4.56s / frame 114",
      mean_ms: 938.6,
      median_ms: 929.1,
      min_ms: 814.3,
      max_ms: 1037.3
    },
    {
      task: "place_dual_shoes",
      forced_threshold_s: 2.61,
      actual_trigger: "2.64s / frame 66",
      mean_ms: 1041.2,
      median_ms: 1025.9,
      min_ms: 991.5,
      max_ms: 1099.5
    }
  ]
};
