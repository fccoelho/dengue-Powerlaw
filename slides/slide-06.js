const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 6,
  title: 'Alpha Distribution'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Distribution of α Parameters", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.95, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addImage({
    x: 0.5, y: 1.2, w: 5.5, h: 4.0,
    path: "./imgs/alpha_distribution.png"
  });

  const statsBox = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.3, y: 1.2, w: 3.2, h: 2.2,
      fill: { color: "FFFFFF" },
      line: { color: theme.accent, width: 2 },
      rectRadius: 0.1
    });
    slide.addText("Key Statistics", {
      x: 6.5, y: 1.35, w: 3, h: 0.4,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText([
      { text: "Mean α ≈ 2.3", options: { bullet: true, breakLine: true } },
      { text: "Range: 1.5 - 4.0", options: { bullet: true, breakLine: true } },
      { text: "Lower α = heavier tail", options: { bullet: true, breakLine: true } },
      { text: "More extreme events", options: { bullet: true } }
    ], {
      x: 6.5, y: 1.8, w: 2.9, h: 1.5,
      fontSize: 12, fontFace: "Arial",
      color: theme.secondary
    });
  };
  statsBox();

  const interpretation = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.3, y: 3.55, w: 3.2, h: 1.6,
      fill: { color: theme.primary },
      rectRadius: 0.1
    });
    slide.addText("Interpretation", {
      x: 6.5, y: 3.7, w: 3, h: 0.35,
      fontSize: 14, fontFace: "Arial",
      color: "FFFFFF", bold: true
    });
    slide.addText("α ≈ 2 means probability of extreme outbreak (10x typical) is ~1%", {
      x: 6.5, y: 4.1, w: 2.9, h: 0.9,
      fontSize: 11, fontFace: "Arial",
      color: theme.light
    });
  };
  interpretation();

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("6", {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = {
    primary: "1A365D",
    secondary: "2C5282",
    accent: "4ECDC4",
    light: "E2E8F0",
    bg: "F7FAFC"
  };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-06-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
