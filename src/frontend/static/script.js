(() => {
  const taskSelect = document.getElementById("task-select");
  const paramsContainer = document.getElementById("params-container");
  const runButton = document.getElementById("run-button");
  const outputArea = document.getElementById("output-area");
  const statusText = document.getElementById("status-text");

  /** @type {Record<string, {label: string, params: any[], staticFlags: string[]}>} */
  let tasks = {};
  /** @type {HTMLInputElement[]} */
  let paramInputs = [];

  const setStatus = (text, mode = "info") => {
    statusText.textContent = text;
    statusText.classList.remove("error", "success");
    if (mode === "error") {
      statusText.classList.add("error");
    } else if (mode === "success") {
      statusText.classList.add("success");
    }
  };

  const renderParams = (taskData) => {
    paramsContainer.innerHTML = "";
    paramInputs = [];

    if (!taskData || !taskData.params || taskData.params.length === 0) {
      paramsContainer.innerHTML =
        '<div class="empty-hint">No configurable parameters for this task</div>';
      return;
    }

    taskData.params.forEach((param) => {
      const wrapper = document.createElement("div");
      wrapper.className = "param-field";

      const label = document.createElement("label");
      label.className = "param-label";
      label.textContent = param.label;
      label.setAttribute("for", `param-${param.id}`);

      const flag = document.createElement("div");
      flag.className = "param-flag";
      flag.textContent = param.flag;

      const input = document.createElement("input");
      input.type = "text";
      input.id = `param-${param.id}`;
      input.name = param.id;
      input.value = param.default ?? "";
      input.placeholder = param.placeholder ?? "";
      input.className = "input-field";
      input.dataset.paramId = param.id;

      wrapper.appendChild(label);
      wrapper.appendChild(flag);
      wrapper.appendChild(input);
      paramsContainer.appendChild(wrapper);
      paramInputs.push(input);
    });

    if (Array.isArray(taskData.staticFlags) && taskData.staticFlags.length) {
      const hint = document.createElement("div");
      hint.className = "empty-hint";
      hint.textContent = `Static flags: ${taskData.staticFlags.join(" ")}`;
      paramsContainer.appendChild(hint);
    }
  };

  const loadTasks = async () => {
    try {
      const response = await fetch("/tasks");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      tasks = await response.json();
      taskSelect.innerHTML =
        '<option value="">Select a task</option>' +
        Object.entries(tasks)
          .map(
            ([name, info]) =>
              `<option value="${name}">${info.label} (${name})</option>`,
          )
          .join("");
      taskSelect.disabled = false;
      setStatus("Please select a task");
    } catch (error) {
      console.error(error);
      taskSelect.innerHTML =
        '<option value="">Failed to load tasks, refresh to retry</option>';
      setStatus("Failed to load task list", "error");
    }
  };

  const onTaskChange = () => {
    const selected = taskSelect.value;
    if (!selected || !tasks[selected]) {
      paramsContainer.innerHTML =
        '<div class="empty-hint">Select a task to view and edit parameters</div>';
      paramInputs = [];
      runButton.disabled = true;
      setStatus("Please select a task");
      return;
    }
    renderParams(tasks[selected]);
    runButton.disabled = false;
    setStatus(`Selected task: ${tasks[selected].label}`);
  };

  const runTask = async () => {
    const task = taskSelect.value;
    if (!task || !tasks[task]) {
      setStatus("Please select a task", "error");
      return;
    }

    const params = {};
    paramInputs.forEach((input) => {
      params[input.dataset.paramId] = input.value;
    });

    runButton.disabled = true;
    taskSelect.disabled = true;
    setStatus("Task running...");
    outputArea.value = "";

    try {
      const response = await fetch("/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          task,
          params,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = data.error || `Request failed (HTTP ${response.status})`;
        throw new Error(message);
      }

      const header = data.command ? `> ${data.command}\n\n` : "";
      outputArea.value = `${header}${data.output || ""}`;
      setStatus(`Completed, exit code ${data.exitCode}`, "success");
    } catch (error) {
      outputArea.value = "";
      setStatus(`Task failed: ${error.message}`, "error");
    } finally {
      runButton.disabled = false;
      taskSelect.disabled = false;
    }
  };

  taskSelect.addEventListener("change", onTaskChange);
  runButton.addEventListener("click", runTask);

  loadTasks();
})();
