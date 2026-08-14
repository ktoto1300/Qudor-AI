'use strict';
/*
 * Bridge for gorisanson/quoridor-ai (pure-JS MCTS + heuristics).
 *
 * Runs in Node by concatenating the repo's game.js + ai.js into one vm context,
 * then answering our JSON-lines protocol. Their move triple:
 *   [[row, col], null, null] pawn destination (row/col from TOP, 0..8)
 *   [null, [row, col], null] horizontal wall slot (0..7)  == our 81+r*8+c
 *   [null, null, [row, col]] vertical   wall slot (0..7)  == our 145+r*8+c
 * Same wall geometry as our engine (h wall covers columns c..c+1, v wall rows r..r+1).
 */
const fs = require('fs');
const readline = require('readline');
const vm = require('vm');
const path = require('path');

const REPO = path.join(__dirname, '..', '..', '..', 'bots', 'gorisanson_quoridor-ai');
const SIMS = parseInt(process.env.GORISANSON_SIMS || '60000', 10);
const UCT = 0.2;

console.log = () => {};               // their AI logs move timing to stdout

function mulberry32(a) {
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

let src = fs.readFileSync(path.join(REPO, 'src', 'js', 'game.js'), 'utf8');
src += '\n' + fs.readFileSync(path.join(REPO, 'src', 'js', 'ai.js'), 'utf8');
vm.runInThisContext(src, { filename: 'gorisanson.js' });

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

rl.on('line', (line) => {
    const msg = JSON.parse(line);
    if (msg.type === 'hello') {
        Math.random = mulberry32((msg.seed || 0) * 2654435761 + 12345);
        process.stdout.write(JSON.stringify({ ok: true, name: 'gorisanson', sims: SIMS, uct: UCT }) + '\n');
        return;
    }
    if (msg.type === 'bye') {
        process.exit(0);
    }
    try {
        const game = new Game(true);               // pawns[0] = light = our p0 (bottom)
        game.board.pawns[0].position = new PawnPosition(Math.floor(msg.p0 / 9), msg.p0 % 9);
        game.board.pawns[1].position = new PawnPosition(Math.floor(msg.p1 / 9), msg.p1 % 9);
        for (const slot of msg.h || []) {
            const r = Math.floor(slot / 8), c = slot % 8;
            game.placeHorizontalWall(r, c, false);
        }
        for (const slot of msg.v || []) {
            const r = Math.floor(slot / 8), c = slot % 8;
            game.placeVerticalWall(r, c, false);
        }
        game.board.pawns[0].numberOfLeftWalls = msg.w0;
        game.board.pawns[1].numberOfLeftWalls = msg.w1;
        game._turn = msg.ply;
        game._validNextPositionsUpdated = false;
        game._probableValidNextWallsUpdated = false;

        const ai = new AI(SIMS, UCT);
        const move = ai.chooseNextMove(game);
        let a = null;
        if (move[0]) {
            const r = move[0][0], c = move[0][1];
            if (game.validNextPositions[r][c]) a = r * 9 + c;
        } else if (move[1]) {
            const r = move[1][0], c = move[1][1];
            if (game.validNextWalls.horizontal[r][c]) a = 81 + r * 8 + c;
        } else if (move[2]) {
            const r = move[2][0], c = move[2][1];
            if (game.validNextWalls.vertical[r][c]) a = 145 + r * 8 + c;
        }
        if (a === null) {
            const tuples = game.getArrOfValidNextPositionTuples();   // safety fallback
            const t = tuples[Math.floor(Math.random() * tuples.length)];
            a = t[0] * 9 + t[1];
        }
        process.stdout.write(JSON.stringify({ a }) + '\n');
    } catch (e) {
        process.stderr.write(String(e && e.stack || e) + '\n');
        process.stdout.write(JSON.stringify({ forfeit: 'gorisanson error: ' + e }) + '\n');
    }
});