import { useState } from 'react'
import MathExplainer from './MathExplainer'
import HistoryTab from './HistoryTab'
import ExplorerGuide from './ExplorerGuide'

const TABS = [
    { id: 'math', label: '📐 Math' },
    { id: 'guide', label: '🧭 Guide' },
    { id: 'history', label: '📜 History' },
]

export default function Sidebar() {
    const [activeTab, setActiveTab] = useState('math')

    return (
        <aside className="sidebar">
            {/* Tab Navigation */}
            <div className="tab-nav glass-card" style={{ padding: 4 }}>
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        className={`tab-btn${activeTab === tab.id ? ' active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                        id={`tab-${tab.id}`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div key={activeTab} className="tab-content">
                {activeTab === 'math' && <MathExplainer />}
                {activeTab === 'guide' && <ExplorerGuide />}
                {activeTab === 'history' && <HistoryTab />}
            </div>
        </aside>
    )
}
