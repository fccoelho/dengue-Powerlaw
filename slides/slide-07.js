const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 7,
  title: 'Spatial Patterns'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Spatial Patterns: α by State", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.95, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addImage({
    x: 0.3, y: 1.1, w: 6.0, h: 4.3,
    path: "./imgs/alpha_by_state.png"
  });

  const insights = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.5, y: 1.1, w: 3.0, h: 4.3,
      fill: { color: "FFFFFF" },
      line: { color: theme.accent, width: 2 },
      rectRadius: 0.1
    });
    slide.addText("Regional Insights", {
      x: 6.7, y: 1.25, w: 2.8, h: 0.4,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText([
      { text: "North & Northeast", options: { bold: true, breakLine: true } },
      { text: "Lower α → heavier tails\nMore extreme outbreaks", options: { breakLine: true, color: theme.secondary } },
      { text: "\nSoutheast", options: { bold: true, breakLine: true } },
      { text: "Higher α → thinner tails\nMore predictable", options: { breakLine: true, color: theme.secondary } },
      { text: "\nAmazon Region", options: { bold: true, breakLine: true } },
      { text: "Variable α values\nHeterogeneous dynamics", options: { color: theme.secondary } }
    ], {
      x: 6.7, y: 1.7, w: 2.6, h: 3.5,
      fontSize: 11, fontFace: "Arial",
      color: theme.primary
    });
  };
  insights();

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("7", {
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
  pres.writeFile({ fileName: "slide-07-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
