const fsw = require('@sutton-signwriting/core/fsw')

function parse(raw, includeSpatials=true) {
    let { sequence, spatials } = fsw.parse.sign(raw)
    sequence = sequence.join(' ')
    spatials = spatials.map(x => x.coord.join(' ')).join(' ')
    if (includeSpatials) {
        sequence = sequence + ' ' + spatials
    }
    return sequence
}

const result = parse(process.argv[2])
// const result = parse('AS20310S26b02S33100M521x547S33100482x483S20310506x500S26b02503x520')

console.log(result)