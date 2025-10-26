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
        '<div class="empty-hint">该任务没有可配置的参数</div>';
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
      hint.textContent = `固定启用的标志：${taskData.staticFlags.join(" ")}`;
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
        '<option value="">请选择任务</option>' +
        Object.entries(tasks)
          .map(
            ([name, info]) =>
              `<option value="${name}">${info.label} (${name})</option>`,
          )
          .join("");
      taskSelect.disabled = false;
      setStatus("请选择任务");
    } catch (error) {
      console.error(error);
      taskSelect.innerHTML =
        '<option value="">加载失败，请刷新页面重试</option>';
      setStatus("任务列表加载失败", "error");
    }
  };

  const onTaskChange = () => {
    const selected = taskSelect.value;
    if (!selected || !tasks[selected]) {
      paramsContainer.innerHTML =
        '<div class="empty-hint">请选择任务以查看并编辑参数</div>';
      paramInputs = [];
      runButton.disabled = true;
      setStatus("请选择任务");
      return;
    }
    renderParams(tasks[selected]);
    runButton.disabled = false;
    setStatus(`已选择任务：${tasks[selected].label}`);
  };

  const runTask = async () => {
    const task = taskSelect.value;
    if (!task || !tasks[task]) {
      setStatus("请选择任务", "error");
      return;
    }

    const params = {};
    paramInputs.forEach((input) => {
      params[input.dataset.paramId] = input.value;
    });

    runButton.disabled = true;
    taskSelect.disabled = true;
    setStatus("任务运行中…");
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
        const message = data.error || `请求失败 (HTTP ${response.status})`;
        throw new Error(message);
      }

      const header = data.command ? `> ${data.command}\n\n` : "";
      outputArea.value = `${header}${data.output || ""}`;
      setStatus(`完成，退出码 ${data.exitCode}`, "success");
    } catch (error) {
      outputArea.value = "";
      setStatus(`运行失败：${error.message}`, "error");
    } finally {
      runButton.disabled = false;
      taskSelect.disabled = false;
    }
  };

  taskSelect.addEventListener("change", onTaskChange);
  runButton.addEventListener("click", runTask);

  loadTasks();
})();
