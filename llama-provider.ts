const cache_directory = "./cache"; // Stores all cached data.
const llama_src_path = `${cache_directory}/llama.cpp.zip`;
const llama_root_folder = "llama.cpp-master";
const llama_extract_path = `${cache_directory}/llama-src/`; // the zip has a root directory, so extract here
const llama_build_path = `${cache_directory}/llama-build`;
const gguf_directory = "./models";
const llamaCPPSourceUrl = `https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip`;

const llamaServerExecutablePath = `${llama_build_path}/bin/llama-server`;

import * as path from "path";

let context_size: number = 12288; // defaults to a 12k context window, system will try and scale up if overflows occur.
let context_scales: number = 0; // number of times we've had to scale up the context window due to overflows.

let inferenceServer: llamaServer | null = null;
let embedServer: llamaServer | null = null;

let meushi: any = null;
let configManager: any = null;
let config: any = null;

// Bun.spawn wrapper that logs output via meushi.log
async function runCmd( args: string[], options: Record<string, any> = {} ): Promise<void> {
    const proc = Bun.spawn(args, {
        stdout: "pipe",
        stderr: "pipe",
        ...options,
    });

    // Stream stdout and stderr to meushi.log
    const streamToLog = async ( stream: ReadableStream<Uint8Array> | null, level: string ) => {
        if (!stream) return;
        const reader = stream.getReader();
        const decoder = new TextDecoder();
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                if (value) {
                    const text = decoder.decode(value, { stream: true });
                    meushi.log(level, text.trimEnd());
                }
            }
        } catch (e) {
            // Stream closed, ignore
        }
    };

    // Run stream readers in parallel, don't await them before waiting for exit
    const stdoutPromise = streamToLog(proc.stdout, "info");
    const stderrPromise = streamToLog(proc.stderr, "error");

    const exitCode = await proc.exited;

    // Wait for streams to finish
    await Promise.all([stdoutPromise, stderrPromise]);

    if (exitCode !== 0) {
        throw new Error(
            `Command '${args[0]}' failed with exit code ${exitCode}`,
        );
    }
}

async function downloadFile(url: string, dest: string): Promise<void> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    await Bun.write(dest, new Uint8Array(buffer));
}

function scanOpenPorts( min: number, max: number, maxResults: number = 5 ): number[] {
    const openPorts: number[] = [];

    let workingPort = min;
    while (openPorts.length < maxResults && workingPort <= max) {
        // Attempt to bind to the port to check if it's open
        try {
            const server = Bun.serve({
                port: workingPort,
                fetch() {
                    return new Response("OK");
                },
            });
            server.stop(true); // Immediately stop the server, we just wanted to check the port
            openPorts.push(workingPort);
        } catch (e) {
            // Port is in use, ignore and continue scanning
        }

        workingPort++;

        if (workingPort > max) {
            return openPorts; // Reached the end of the range
        }
    }

    return openPorts;
}

async function buildFileSystem() {
    building = true;
    // Ensure cache directories
    await runCmd([
        "mkdir",
        "-p",
        cache_directory,
        llama_extract_path,
        llama_build_path,
        gguf_directory,
    ]);

    // Check if the built-date stub file exists. If not, we assume we need to build.
    const buildDateFile = path.join(cache_directory, "llama-built-date");
    let needsBuild = !(await Bun.file(buildDateFile).exists());

    if (!needsBuild) {
        // Read unix timestamp in the file and check if it's older than the expiry threshold.
        const builtTimestampStr = await Bun.file(buildDateFile).text();
        const builtTimestamp = parseInt(builtTimestampStr, 10);
        const now = Date.now();
        const ageDays = (now - builtTimestamp) / (1000 * 60 * 60 * 24);

        if (ageDays > 30) {
            // If the build is older than 30 days, we rebuild to ensure compatibility and performance.
            meushi.log(
                "info",
                `Existing llama.cpp build is ${Math.floor(ageDays)} days old. Rebuilding...`,
            );
            needsBuild = true;

            // Delete the old build and source to ensure a clean slate. We will redownload and rebuild.
            await runCmd(["rm", "-rf", llama_build_path, llama_extract_path]);
        } else {
            meushi.log(
                "info",
                `Existing llama.cpp build is ${Math.floor(ageDays)} days old. No rebuild needed.`,
            );
            building = false;
            return; // No need to build, exit early.
        }
    }

    // Check if llama.cpp source zip exists
    if (!(await Bun.file(llama_src_path).exists())) {
        meushi.log("info", "Downloading llama.cpp source zip...");
        await downloadFile(llamaCPPSourceUrl, llama_src_path);
    } else {
        meushi.log(
            "info",
            "llama.cpp source zip already exists, skipping download.",
        );
    }

    // Extract source, overwrite if already exists.
    meushi.log("info", "Extracting llama.cpp source...");
    await runCmd(["unzip", "-o", llama_src_path, "-d", llama_extract_path], {
        stdout: "inherit",
        stderr: "inherit",
    });

    // Build llama.cpp
    meushi.log(`Building llama.cpp...`);
    await runCmd(
        [
            "cmake",
            "-B",
            llama_build_path,
            "-S",
            `${llama_extract_path}${llama_root_folder}`,
            "-DGGML_CUDA=ON",
        ],
        { stdout: "inherit", stderr: "inherit" },
    );
    await runCmd(
        ["cmake", "--build", llama_build_path, "--config", "Release"],
        { stdout: "inherit", stderr: "inherit" },
    );

    meushi.log("llama.cpp build complete!");

    // Write the current timestamp to the built-date file to track when we built. This will help us determine when to rebuild in the future.
    // if file exists, overwrite it with the current timestamp. If it doesn't exist, create it.
    await Bun.write(buildDateFile, Date.now().toString());
    building = false;
}

async function checkModelExists(filename: string): Promise<boolean> {
    return await Bun.file(gguf_directory + "/" + filename).exists();
}

class llamaServer {
    modelPath: string;
    process: any;
    running: boolean;
    port: number;
    address: string;

    constructor(modelPath: string, address: string, port: number) {
        this.modelPath = modelPath;
        this.process = null;
        this.running = false;
        this.port = port || 8080;
        this.address = address || "localhost";
    }

    async start(isEmbedding: boolean = false) {
        if (this.running) {
            console.log("Server is already running.");
            return;
        }

        console.log(
            `Starting llama-server with model ${this.modelPath} on ${this.address}:${this.port}...`,
        );

        let sequence = [
            llamaServerExecutablePath,
            "-m",
            this.modelPath,
            "-c",
            "12298",
            "--n-gpu-layers",
            "99",
            "--port",
            this.port.toString(),
            "--host",
            this.address,
            "--cont-batching",
        ];

        if (isEmbedding === true) {
            sequence.push("--embeddings");
        }

        this.process = Bun.spawn(sequence, {
            stdio: ["ignore", "inherit", "inherit"],
        });

        await this.waitForReady();
        this.running = true;
        console.log("llama-server started!");
    }

    async stop() {
        if (!this.running) {
            console.log("Server is not running.");
            return;
        }

        console.log("Stopping llama-server...");
        this.process.kill("SIGINT");
        await this.process.exited;
        this.running = false;
        console.log("llama-server stopped.");
    }

    private async waitForReady() {
        console.log("Waiting for llama-server to be ready...");
        const start = Date.now();
        while (Date.now() - start < 180000) {
            // wait up to 3 minutes
            try {
                const res = await fetch(
                    `http://${this.address}:${this.port}/health`,
                );
                if (res.ok && (await res.json()).status === "ok") {
                    console.log("llama-server is ready!");
                    return;
                }
            } catch {}
            await Bun.sleep(500);
        }
        throw new Error(
            `llama-server on port ${this.port} failed to start within 3 minutes`,
        );
    }

    // Writes logs to cache/logs/index.txt. where index is the next available index number. Each log entry is a JSON object with {timestamp, systemPrompt, userPrompt, response}.
    private writeLog(
        systemPrompt: string,
        userPrompt: string,
        response: string,
    ) {
        const fileID = Date.now(); // Use timestamp as unique ID for log file
        const logEntry = {
            timestamp: new Date().toISOString(),
            systemPrompt,
            userPrompt,
            response,
        };
        const logPath = path.join(cache_directory, "logs", `${fileID}.json`);
        Bun.write(logPath, JSON.stringify(logEntry, null, 2));
    }

    // Generates a single response.
    async generate(systemPrompt: string, userPrompt: string): Promise<string> {
        let tryAgain: boolean = true;
        let tries = 0;
        while(tryAgain) {
            try {
                if (!this.running) {
                    return "Server is not running. Please start the server before generating responses.";
                }

                const res = await fetch(
                    `http://${this.address}:${this.port}/v1/chat/completions`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            model: "local-model",
                            messages: [
                                { role: "system", content: systemPrompt },
                                { role: "user", content: userPrompt },
                            ],
                            stream: false,
                            temperature: 0.7,
                            max_tokens: 2024,
                            top_p: 0.95,
                        }),
                    },
                );

                if (!res.ok) {
                    const errorText = await res.text();
                    throw new Error(`Error from llama-server: ${errorText}`);
                }

                const data = await res.json();

                // Write to log
                this.writeLog(
                    systemPrompt,
                    userPrompt,
                    data.choices[0].message.content,
                );

                return (tries > 0) ? data.choices[0].message.content + ` (Generated in ${tries} tries)` : data.choices[0].message.content;
            } catch (err) {
                meushi.error("Error generating response:", err);
                
                // Look for error code 500 and "Context size has been exceeded." text.
                if(err.message.includes("500") && err.message.includes("exceeded") && context_scales < 5) {
                    // If we hit a context size overflow, we increase the context window and try again.
                    context_scales++;
                    context_size += 1024; // Increase context by 1k tokens
                    tries++; // Track the number of tries we've made to increase context.
                } else {
                    return `Error generating response: ${err.message}`;
                }
            }
        }
    }

    async embed(text: string): Promise<number[]> {
        if (!this.running) {
            throw new Error(
                "Server is not running. Please start the server before generating embeddings.",
            );
        }

        const res = await fetch(
            `http://${this.address}:${this.port}/v1/embeddings`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: "local-model",
                    input: text,
                }),
            },
        );

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Error from llama-server: ${errorText}`);
        }

        const data = await res.json();
        return data.data[0].embedding;
    }
}

async function init(source: any) {
    try {
        meushi = source;

        // Register exit handlers early to ensure cleanup on crash/exit
        registerExitHandlers();

        configManager = new meushi.configRegisterClass("llama-provider.conf");
        configManager.registerConfig(
            "INFERENCE_MODEL",
            "mistral-nemo.gguf",
            "The .gguf model file to use for inference (must be in the models/ directory)",
        );
        configManager.registerConfig(
            "EMBED_MODEL",
            "nomic_embed_text.gguf",
            "The .gguf model file to use for embeddings (must be in the models/ directory)",
        );
        config = await configManager.getConfig();

        await buildFileSystem();

        const modelFiles = fs
            .readdirSync(gguf_directory)
            .filter((file) => file.endsWith(".gguf"));
        if (modelFiles.length === 0) {
            throw new Error(
                `No .gguf model files found in ${gguf_directory}. Please add a model and try again.`,
            ); // Modulize will cleanly handle this.
        }

        if (
            !modelFiles.includes(config.INFERENCE_MODEL) ||
            !modelFiles.includes(config.EMBED_MODEL)
        ) {
            meushi.log(
                `Configured models not found in ${gguf_directory}. Available models: ${modelFiles.join(", ")}`,
            );
            meushi.log(
                `Please add the configured models or update the config to match available models.`,
            );
            meushi.log(
                `Expected models - Inference: ${config.INFERENCE_MODEL}, Embedding: ${config.EMBED_MODEL}`,
            );
            throw new Error(
                "Configured model(s) not found. Check logs for details.",
            );
        }

        // We know we have the files because of the check above, so we can directly start the servers without awaiting the file existence.
        // First get some open ports.
        let portsAvailable: number[] = scanOpenPorts(4000, 5000, 2);
        if (portsAvailable.length < 2) {
            throw new Error(
                "Not enough open ports available for llama-server. Please free up some ports between 4000-5000 and try again.",
            );
        }

        inferenceServer = new llamaServer(
            path.join(gguf_directory, config.INFERENCE_MODEL),
            "localhost",
            portsAvailable[0],
        );
        embedServer = new llamaServer(
            path.join(gguf_directory, config.EMBED_MODEL),
            "localhost",
            portsAvailable[1],
        );

        await inferenceServer.start();
        await embedServer.start(true);

        meushi.log("llama-server component initialized successfully!");
    } catch (err) {
        throw new Error(
            `Failed to initialize llama-provider component: ${err.message}`,
        );
    }
}

async function generate(systemPrompt: string, prompt: string): Promise<string> {
    if (!inferenceServer) {
        throw new Error("Inference server is not initialized.");
    }

    return await inferenceServer.generate(systemPrompt, prompt);
}

async function embed(text: string): Promise<number[]> {
    if (!embedServer) {
        throw new Error("Embedding server is not initialized.");
    }

    return await embedServer.embed(text);
}

async function shutdown() {
    await inferenceServer?.stop();
    await embedServer?.stop();
    inferenceServer = null;
    embedServer = null;
}

// Register process exit handlers to clean up llama-server processes
function registerExitHandlers() {
    const cleanup = () => {
        console.log("[llama-provider] Cleaning up servers on exit...");
        inferenceServer?.process?.kill("SIGTERM");
        embedServer?.process?.kill("SIGTERM");
    };

    process.on("exit", cleanup);
    process.on("SIGINT", () => {
        cleanup();
        process.exit(130);
    });
    process.on("SIGTERM", () => {
        cleanup();
        process.exit(143);
    });
    process.on("uncaughtException", (err) => {
        console.error("[llama-provider] Uncaught exception:", err);
        cleanup();
        process.exit(1);
    });
}

export {
    init,
    generate,
    embed,
    shutdown
}
