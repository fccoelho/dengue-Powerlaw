const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'toc',
  index: 2,
  title: 'Overview'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Overview", {
    x: 0.5, y: 0.3, w: 9, h: 0.8,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.0, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  const topics = [
    { num: "01", title: "The Challenge", desc: "Why power laws matter for epidemics" },
    { num: "02", title: "Data & Methods", desc: "Brazilian dengue surveillance data" },
    { num: "03", title: "Power Law Fitting", desc: "Statistical methodology" },
    { num: "04", title: "Key Findings", desc: "Spatial and temporal patterns" },
    { num: "05", title: "Machine Learning", desc: "Predictive modeling with α features" },
    { num: "06", title: "Implications", desc: "Early warning & resource allocation" }
  ];

  topics.forEach((topic, i) => {
    const y = 1.4 + i * 0.65;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.7, y: y + 0.05, w: 0.4, h: 0.4,
      fill: { color: theme.accent }
    });
    slide.addText(topic.num, {
      x: 0.7, y: y + 0.05, w: 0.4, h: 0.4,
      fontSize: 11, fontFace: "Arial",
      color: "FFFFFF", bold: true,
      align: "center", valign: "middle"
    });
    slide.addText(topic.title, {
      x: 1.3, y: y, w: 3, h: 0.3,
      fontSize: 16, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText(topic.desc, {
      x: 1.3, y: y + 0.28, w: 4, h: 0.3,
      fontSize: 12, fontFace: "Arial",
      color: theme.secondary
    });
  });

  slide.addImage({
    x: 6.0, y: 1.5, w: 3.5, h: 3.5,
    path: "./imgs/alpha_distribution.png"
  });

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("2", {
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
  pres.writeFile({ fileName: "slide-02-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
