const config = require('./nconfig.json');
const fs = require('fs');
const colors = require('colors');
const moment = require('moment');
var momentDurationFormatSetup = require("moment-duration-format");
const readline = require('readline');
const { spawn } = require('child_process');


Array.prototype.max = function () {
    return Math.max.apply(null, this);
};

function* getFileName() {
    files = fs.readdirSync(`${__dirname}/${config.dir}`);

    var index = -1;
    
    while (index < files.length) {
        index++;
        console.log(`Current number: ${index}`);
        
        yield files[index];
    }
}

var getter = getFileName();
let subprocess = [];

async function run(params, id) { 
    return new Promise((resolve, reject) => {
        let env = JSON.parse(JSON.stringify(process.env));
        env['CUDA_VISIBLE_DEVICES'] = Math.floor(id / params.workers_on_gpu);

        let args = [];
        args.push(params.file);

        let item = getter.next();
        if (item.done)
            resolve(null)

        let filename = item.value
        args.push(`--filename=${filename}`);
        // args.push(`--tailnumber=${ params.tailnumber }`);
        // args.push(`--workers=${ config.workers }`);

        let startedTime = moment();
        console.log(`${filename.magenta}: started ${startedTime.format("kk:mm DD.MM.YY").green}`);
        const command = spawn('python', args, { cwd: params.path, env: env });

        command.stdout.on('data', async (data) => {
            try {
                // console.log(`${filename}: ${data}`);

            } catch (error) {
                console.error(error.msg);
            }
        });

        command.stderr.on('data', (data) => {
            // console.log(`${filename}: ${data.toString()}`);
        });

        command.on('close', (code) => {
            console.log(`${filename} ended with code ${code}`);
            resolve(id);
        });

        command.on('error', (err) => {
            console.log(`${err}`);
        });

        subprocess[id] = command;
    });
}

let workers = config.num_gpu * config.workers_on_gpu;

async function start(id) {
    id = await run(config, id);

    while (id) {
        id = await run(config, id);
    }
}

for (let i = 0; i < workers; i++) {
    try {
        start(i);
    } catch (error) {
        console.log(error);
    }
}

readline.emitKeypressEvents(process.stdin);

if (process.stdin.isTTY) {
    process.stdin.setRawMode(true);
}

const listener = (ch, key) => {
    if (key && key.name === 'escape') {
        subprocess.forEach((sub) => {
            sub.kill();
        });

        process.exit();
    }
};

process.stdin.on('keypress', listener);

